"""Gymnasium 1v1 melee env. Agent = player 0, opponent drives player 1."""
from __future__ import annotations

import math

import numpy as np
import gymnasium as gym

from sandbox.config import CombatConfig, PhysicsConfig, SimConfig, WorldConfig
from sandbox.simulation.sim import PvPSim, idle_ctrl
from rl.environment import action_map, obs_schema
from rl.opponents.scripted import create as create_opponent
from rl.rewards.shaping import RewardBook, RewardConfig

try:
    from replay.recorder import ReplayRecorder
except Exception:  # headless / partial checkout
    ReplayRecorder = None


class PvPEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}

    def __init__(self, opponent: str = "dummy", opponent_kwargs: dict | None = None,
                 reward_cfg: RewardConfig | None = None,
                 world_cfg: WorldConfig | None = None,
                 phys_cfg: PhysicsConfig | None = None,
                 combat_cfg: CombatConfig | None = None,
                 max_ticks: int = 900, frame_skip: int = 4,
                 record: bool = False, render_mode=None, seed: int = 0,
                 obs_version: int = obs_schema.OBS_VERSION,
                 loadout: dict | None = None,
                 enabled_items: set | list | None = None,
                 mirror: bool = False,
                 agent_start_hp: list | None = None,
                 foe_start_hp: list | None = None,
                 start_dist: float | None = None):
        super().__init__()
        self.opponent_name = opponent
        self.opponent_kwargs = opponent_kwargs or {}
        self.reward_cfg = reward_cfg or RewardConfig()
        self._world_cfg = world_cfg or WorldConfig()
        self._phys_cfg = phys_cfg or PhysicsConfig()
        self._combat_cfg = combat_cfg or CombatConfig()
        self.max_ticks = max_ticks
        self.frame_skip = frame_skip
        self.record = record and ReplayRecorder is not None
        self.render_mode = render_mode
        self.action_space = action_map.action_space()
        self.observation_space = obs_schema.obs_space(obs_version)
        self.obs_version = obs_version
        self.sim = PvPSim(self._world_cfg, self._phys_cfg, self._combat_cfg,
                          max_ticks=max_ticks, seed=seed,
                          loadout=loadout,
                          enabled_items=set(enabled_items) if enabled_items else None)
        self.mirror = mirror
        self.agent_start_hp = agent_start_hp
        self.foe_start_hp = foe_start_hp
        self.start_dist = start_dist
        self.foe_pool: list | None = None
        self._foe_rng = np.random.default_rng(seed ^ 0x9E3779B9)
        self.opponent = create_opponent(opponent, **self.opponent_kwargs)
        self.book = RewardBook(self.reward_cfg)
        self.recorder = None
        self._last_action = np.zeros(6, dtype=int)
        self._last_comp: dict = {}

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.sim.reset(seed, mirror=self.mirror)
            self.np_random = np.random.default_rng(seed)
            self._draw_foe(seed)
            if not self.foe_pool:
                self.opponent.reset(seed)
            if self.agent_start_hp is not None:
                lo, hi = self.agent_start_hp
                hp = float(self.sim.rng.uniform(lo, hi))
                self.sim.players[0].health = min(self.sim.players[0].max_health, hp)
            self._apply_scenario()
        else:
            self.sim.reset(mirror=self.mirror)
            self._draw_foe(None)
            if not self.foe_pool:
                self.opponent.reset(int(self.sim.rng.integers(0, 2**31 - 1)))
            if self.agent_start_hp is not None:
                lo, hi = self.agent_start_hp
                hp = float(self.sim.rng.uniform(lo, hi))
                self.sim.players[0].health = min(self.sim.players[0].max_health, hp)
            self._apply_scenario()
        self.book.reset()
        if self.record:
            from replay.recorder import ReplayRecorder as RR
            self.recorder = RR()
            self.recorder.begin(self.sim)
        obs = obs_schema.build_obs(self.sim, agent=0, version=self.obs_version)
        return obs.astype(np.float32), self._info(done=False)

    def step(self, action):
        ctrl0 = action_map.to_controller(action)
        self._last_action = np.asarray(action).flatten().copy()
        total, comp_sum = 0.0, None
        comp_acc = None
        events_all = []
        for _ in range(self.frame_skip):
            if self.sim.done:
                break
            ctrl1 = self.opponent.act(self.sim, 1)
            ev = self.sim.step(ctrl0, ctrl1)
            events_all.extend(ev["events"])
            facing = self._facing(0)
            r, comp = self.book.step(self.sim, agent=0, done=self.sim.done, facing=facing)
            total += r
            comp_acc = comp if comp_acc is None else {k: comp_acc[k] + comp[k] for k in comp}
            if self.recorder is not None:
                self.recorder.capture(self.sim, ctrl0, ctrl1, r, comp)
        self._last_comp = comp_acc or {}
        obs = obs_schema.build_obs(self.sim, agent=0, version=self.obs_version)
        done = self.sim.done
        terminated = bool(done and (self.sim.winner is not None or self.sim.invalid))
        truncated = bool(done and not terminated)
        info = self._info(done=done)
        info["reward_components"] = self._last_comp
        if self.recorder is not None and done:
            self.recorder.finish(self.sim)
        return obs.astype(np.float32), float(total), terminated, truncated, info

    # -- helpers -------------------------------------------------------------
    def _apply_scenario(self) -> None:
        """Headline-scenario overrides. All None by default (legacy behavior
        bit-identical). foe HP + fixed start distance are scenario parameters
        like agent_start_hp, not mechanics."""
        if self.foe_start_hp is not None:
            lo, hi = self.foe_start_hp
            hp = float(self.sim.rng.uniform(lo, hi))
            self.sim.players[1].health = min(self.sim.players[1].max_health, hp)
        if self.start_dist is not None:
            d = float(self.start_dist)
            cx = float(self.sim.world.size) * 0.5
            self.sim.players[0].pos[:] = (cx - d / 2, 0.0, cx)
            self.sim.players[1].pos[:] = (cx + d / 2, 0.0, cx)
            self.sim.players[0].vel[:] = 0.0
            self.sim.players[1].vel[:] = 0.0
            if self.mirror:
                p0 = self.sim.players[0].pos.copy()
                self.sim.players[0].pos[:] = self.sim.players[1].pos
                self.sim.players[1].pos[:] = p0
                self.sim.players[0].yaw, self.sim.players[1].yaw = 180.0, 0.0
            else:
                self.sim.players[0].yaw, self.sim.players[1].yaw = 0.0, 180.0
            for i in (0, 1):
                self.sim.prev_pos[i] = [self.sim.players[i].pos.copy()
                                        for _ in range(4)]

    def _facing(self, agent: int) -> bool:
        # 3D look-vs-target check (same geometry as combat arc): pitching away
        # from the foe must NOT count as facing. (Prevents pitch-down exploits.)
        me, foe = self.sim.players[agent], self.sim.players[1 - agent]
        d = (foe.pos + np.array([0.0, 0.9, 0.0])) - (me.pos + np.array([0.0, 0.9, 0.0]))
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return True
        yr, pr = math.radians(me.yaw), math.radians(me.pitch)
        look = np.array([math.cos(yr) * math.cos(pr), math.sin(pr), math.sin(yr) * math.cos(pr)])
        return float(np.dot(look, d / n)) > 0.5

    def _info(self, done: bool) -> dict:
        s = self.sim
        atk, hit = s.attempts[0], s.hits[0]
        return {
            "tick": s.tick, "done": done, "winner": s.winner,
            "win": 1 if s.winner == 0 else 0, "loss": 1 if s.winner == 1 else 0,
            "distance": s.distance(),
            "damage_dealt": s.damage_dealt[0], "damage_taken": s.damage_dealt[1],
            "hp": s.players[0].health, "foe_hp": s.players[1].health,
            "kills": 1 if s.winner == 0 else 0, "deaths": 1 if s.winner == 1 else 0,
            "attacks": atk, "hits": hit,
            "hit_rate": (hit / atk) if atk else 0.0,
            "attacks_in_range": s.attempts_in_range[0],
            "kb_dealt": s.kb_dealt[0], "kb_taken": s.kb_dealt[1],
            "blocks": s.blocks[0], "blocked_taken": s.blocks[1],
            "items_used": s.items_used[0],
            "foe_attacks": s.attempts[1], "foe_hits": s.hits[1],
            "reward_total": sum(self.book.totals.values()),
            "reward_components": dict(self.book.totals),
            "obs_version": self.obs_version,
            "last_action": self._last_action.copy(),
        }

    def set_opponent(self, name: str, kwargs: dict | None = None, seed: int = 0) -> None:
        """Swap the foe mid-training (curriculum stages). Picklable args only."""
        self.opponent_name = name
        self.opponent_kwargs = kwargs or {}
        self.opponent = create_opponent(name, **self.opponent_kwargs)
        self.opponent.reset(seed)

    def set_enabled(self, items: list | set) -> None:
        """Swap the unlock set mid-training (utility curriculum stages)."""
        items = set(items)
        self.sim.enabled_items = items
        for inv in self.sim.inventories:
            inv.enabled = set(items)

    def set_loadout(self, loadout: dict) -> None:
        """Swap inventory counts mid-training (resource stages)."""
        self.sim.loadout = dict(loadout)
        for inv in self.sim.inventories:
            sel = inv.selected
            inv.reset(self.sim.loadout, set(inv.enabled))
            inv.selected = sel

    def set_regen(self, rate: float = 0.0, delay: int = 0) -> None:
        """Set foe regeneration (attrition scenarios)."""
        self.sim.foe_regen_rate = float(rate)
        self.sim.foe_regen_delay = int(delay)

    def set_obstacles(self, obstacles: list) -> None:
        """Swap static obstacle layout mid-training (terrain stages)."""
        self._world_cfg.obstacles = tuple(tuple(o) for o in obstacles)
        self.sim.world.set_obstacles(self._world_cfg.obstacles)

    def set_foe_pool(self, pool: list | None, seed: int = 0) -> None:
        """Per-episode sampled foe pool: [{opponent, kwargs, weight}].
        None restores fixed-opponent mode. Deterministic via reseeded rng."""
        self.foe_pool = pool
        self._foe_rng = np.random.default_rng(seed ^ 0x9E3779B9)

    def _draw_foe(self, seed: int | None) -> None:
        if not self.foe_pool:
            return
        if seed is not None:
            rng = np.random.default_rng((seed ^ 0xABCD) % (2**31))
        else:
            rng = self._foe_rng
        names = [e.get("opponent", "melee") for e in self.foe_pool]
        ws = np.array([float(e.get("weight", 1.0)) for e in self.foe_pool])
        ws = ws / ws.sum()
        i = int(rng.choice(len(names), p=ws))
        entry = self.foe_pool[i]
        foe_seed = seed if seed is not None else int(rng.integers(0, 2**31 - 1))
        self.set_opponent(entry.get("opponent", "melee"),
                          entry.get("kwargs", {}), foe_seed)
        self._apply_entry_spawns(entry, rng)

    def _apply_entry_spawns(self, entry: dict, rng) -> None:
        """Per-pool-entry spawn overrides (terrain scenarios, e.g. foe on a
        platform + agent at its base). Optional keys agent_spawn/foe_spawn
        ([x, y, z]); jitter widens exact geometry. Absent = default spawns."""
        aspawn, fspawn = entry.get("agent_spawn"), entry.get("foe_spawn")
        if aspawn is None and fspawn is None:
            return
        jit = float(entry.get("spawn_jitter", 0.0))
        if aspawn is not None:
            jx = float(rng.uniform(-jit, jit)) if jit else 0.0
            jz = float(rng.uniform(-jit, jit)) if jit else 0.0
            self.sim.players[0].pos[:] = (aspawn[0] + jx, aspawn[1], aspawn[2] + jz)
            self.sim.players[0].vel[:] = 0.0
        if fspawn is not None:
            self.sim.players[1].pos[:] = (fspawn[0], fspawn[1], fspawn[2])
            self.sim.players[1].vel[:] = 0.0
        for i in (0, 1):
            self.sim.prev_pos[i] = [self.sim.players[i].pos.copy()
                                    for _ in range(4)]

    def render(self):
        if self.render_mode != "human":
            return None
        from visualization.debug_view import render_sim
        render_sim(self.sim, reward_comp=self._last_comp, action=self._last_action)

    def save_replay(self, path: str) -> str | None:
        if self.recorder is None:
            return None
        return self.recorder.save(path)


def make_env(opponent="dummy", seed=0, record=False, frame_skip=4,
             max_ticks=900, obs_version=4, **kw):
    def _thunk():
        env = PvPEnv(opponent=opponent, seed=seed, record=record,
                     frame_skip=frame_skip, max_ticks=max_ticks,
                     obs_version=obs_version, **kw)
        return env
    return _thunk
