"""Core deterministic 1v1 simulator. No RL imports here."""
from __future__ import annotations

import numpy as np

from sandbox.config import CombatConfig, PhysicsConfig, SimConfig, WorldConfig
from sandbox.combat.melee import try_attack
from sandbox.combat.projectile import Projectile
from sandbox.entities.player import PlayerState
from sandbox.items import effects as FX
from sandbox.items import use as USE
from sandbox.items.defs import HOTBAR
from sandbox.items.inventory import DEFAULT_LOADOUT, Inventory
from sandbox.physics.engine import step_player
from sandbox.world.arena import VoxelWorld
from sandbox.world.battlefield import BattleField


def idle_ctrl() -> dict:
    return {"move_x": 0.0, "move_z": 0.0, "sprint": False,
            "jump": False, "attack": False, "yaw_delta": 0.0, "pitch_delta": 0.0,
            "block": False, "use_item": False, "select_slot": None}


class PvPSim:
    def __init__(self, world_cfg: WorldConfig | None = None,
                 phys_cfg: PhysicsConfig | None = None,
                 combat_cfg: CombatConfig | None = None,
                 max_ticks: int | None = None, seed: int = 0,
                 loadout: dict | None = None, enabled_items: set | None = None):
        self.world_cfg = world_cfg or WorldConfig()
        self.phys = phys_cfg or PhysicsConfig()
        self.combat = combat_cfg or CombatConfig()
        self.max_ticks = max_ticks if max_ticks is not None else SimConfig().max_ticks
        self.loadout = dict(loadout) if loadout else dict(DEFAULT_LOADOUT)
        self.enabled_items = set(enabled_items) if enabled_items else set(HOTBAR)
        # adversarial sustain: foe-only regeneration (attrition scenarios).
        # (0, 0) = off = all previous behavior bit-identical.
        self.foe_regen_rate: float = 0.0
        self.foe_regen_delay: int = 0
        self.world = VoxelWorld(self.world_cfg.size, self.world_cfg.wall_margin,
                                self.world_cfg.obstacles)
        self.field = BattleField()
        self.world.field = self.field
        self.inventories = [Inventory(), Inventory()]
        self.projectiles: list[Projectile] = []
        self.utility_events: list[dict] = []  # this tick's events (for replay)
        self.items_used = [0, 0]
        self.players = [PlayerState(max_health=self.combat.max_health),
                        PlayerState(max_health=self.combat.max_health)]
        self.rng = np.random.default_rng(int(seed))
        self.seed_value = int(seed)
        self.tick = 0
        self.done = False
        self.winner: int | None = None
        self.invalid = False
        # stats (index 0/1)
        self.damage_dealt = [0.0, 0.0]
        self.hits = [0, 0]
        self.attempts = [0, 0]
        self.attempts_in_range = [0, 0]
        self.kb_dealt = [0.0, 0.0]
        self.blocks = [0, 0]
        self.last_attempt_tick = [-10**9, -10**9]
        self.last_attempt_hit = [0, 0]
        self.prev_pos = [[], []]
        self.combo = [0, 0]
        self.combo_timer = [0, 0]
        self.last_hit_tick = [-10**9, -10**9]
        self.time_since_hit = [10**9, 10**9]
        self.reset(seed)

    # -- lifecycle ---------------------------------------------------------
    def reset(self, seed: int | None = None, mirror: bool = False) -> None:
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(int(seed))
        s = self.world.size
        jx = float(self.rng.uniform(-1.0, 1.0))
        jz = float(self.rng.uniform(-1.0, 1.0))
        p0 = [s * 0.25 + jx, 0.0, s * 0.5 + jz]
        p1 = [s * 0.75 - jx, 0.0, s * 0.5 - jz]
        y0, y1 = 0.0, 180.0
        if mirror:
            # horizontally mirrored scenario: agent starts east facing west.
            # Forces geometry-dependent (not world-side-memorized) play.
            p0, p1 = p1, p0
            y0, y1 = y1, y0
        self.players[0].reset(p0, yaw=y0, health=self.combat.max_health)
        self.players[1].reset(p1, yaw=y1, health=self.combat.max_health)
        for inv in self.inventories:
            inv.reset(self.loadout, self.enabled_items)
        self.field.reset()
        self.projectiles = []
        self.utility_events = []
        self.items_used = [0, 0]
        self.tick = 0
        self.done = False
        self.winner = None
        self.invalid = False
        self.damage_dealt = [0.0, 0.0]
        self.hits = [0, 0]
        self.attempts = [0, 0]
        self.attempts_in_range = [0, 0]
        self.kb_dealt = [0.0, 0.0]
        self.blocks = [0, 0]
        self.last_attempt_tick = [-10**9, -10**9]
        self.last_attempt_hit = [0, 0]
        self.prev_pos = [[], []]
        self.combo = [0, 0]
        self.combo_timer = [0, 0]
        self.last_hit_tick = [-10**9, -10**9]
        self.time_since_hit = [10**9, 10**9]
        for i in (0, 1):
            self.prev_pos[i] = [self.players[i].pos.copy() for _ in range(4)]

    def step(self, ctrl0: dict, ctrl1: dict) -> dict:
        """Advance exactly one physics tick. Deterministic order: P0 then P1."""
        if self.done:
            return self._event_summary()
        ctrls = [ctrl0, ctrl1]
        was_ground = [self.players[0].on_ground, self.players[1].on_ground]
        for i in (0, 1):
            step_player(self.players[i], ctrls[i], self.world, self.phys)
        land_delay = self.combat.landing_attack_delay_ticks
        if land_delay:
            for i in (0, 1):
                p = self.players[i]
                if not was_ground[i] and p.on_ground and p.alive:
                    # touchdown recovery: attack lockout (0 = baseline no-op)
                    p.attack_cooldown = max(p.attack_cooldown, land_delay)
        self.utility_events = []
        self._utility(ctrls)
        self._step_projectiles()
        self.field.tick()
        events = []
        for atk, vic in ((0, 1), (1, 0)):
            if ctrls[atk].get("attack", False):
                ev = try_attack(self.players[atk], self.players[vic], self.combat)
                ev["attacker"] = atk
                events.append(ev)
                if ev["attempted"]:
                    self.attempts[atk] += 1
                    self.last_attempt_tick[atk] = self.tick
                    self.last_attempt_hit[atk] = 1 if ev["hit"] else 0
                if ev["in_range"]:
                    self.attempts_in_range[atk] += 1
                if ev["hit"] and not ev.get("blocked", False):
                    self.damage_dealt[atk] += ev["damage"]
                    self.hits[atk] += 1
                    self.kb_dealt[atk] += ev.get("kb", 0.0)
                if ev.get("blocked", False):
                    self.blocks[vic] += 1
                if ev["attempted"] and not ev["hit"] and not ev.get("blocked", False):
                    # whiffed swing can break a placed block in front (removal)
                    from sandbox.physics.engine import look_dir as _look
                    a = self.players[atk]
                    lk = _look(a.yaw, a.pitch)
                    pt = a.pos + lk * 2.0
                    if self.field.remove_near(float(pt[0]), 0.5, float(pt[2])):
                        self.utility_events.append({"type": "block_break",
                                                    "player": atk})
                if ev["hit"] and not ev.get("blocked", False):
                    self.last_hit_tick[atk] = self.tick
                    if self.combo_timer[atk] > 0:
                        self.combo[atk] += 1
                    else:
                        self.combo[atk] = 1
                    self.combo_timer[atk] = 60
                    self.combo[1 - atk] = 0
                    self.combo_timer[1 - atk] = 0
        for i in (0, 1):
            if self.combo_timer[i] > 0:
                self.combo_timer[i] -= 1
                if self.combo_timer[i] == 0:
                    self.combo[i] = 0
            self.time_since_hit[i] = self.tick - self.last_hit_tick[i]
            self.prev_pos[i].append(self.players[i].pos.copy())
            self.prev_pos[i] = self.prev_pos[i][-4:]
        if self.foe_regen_rate > 0.0 and self.players[1].alive:
            # last_hit_tick[0] = last tick P0 LANDED on the foe (offensive stat)
            quiet = self.tick - self.last_hit_tick[0]
            if quiet > self.foe_regen_delay:
                self.players[1].health = min(
                    self.players[1].max_health,
                    self.players[1].health + self.foe_regen_rate)
        self.tick += 1
        # termination
        a0, a1 = self.players[0].alive, self.players[1].alive
        if not a0 or not a1:
            self.done = True
            if a0 and not a1:
                self.winner = 0
            elif a1 and not a0:
                self.winner = 1
            else:
                self.winner = None  # double KO
        elif self.tick >= self.max_ticks:
            self.done = True
            self.winner = None
        elif not self._valid():
            self.done = True
            self.invalid = True
            self.winner = None
        return {"tick": self.tick, "events": events, "done": self.done,
                "winner": self.winner, "invalid": self.invalid}

    def _utility(self, ctrls: list[dict]) -> None:
        """Shield / slots / item use / effects / hunger / fluids. No reward here."""
        for i in (0, 1):
            p, inv, c = self.players[i], self.inventories[i], ctrls[i]
            if not p.alive:
                p.shield_up = False
                continue
            # slot select
            sel = c.get("select_slot", None)
            if sel is not None and inv.select(int(sel)):
                self.utility_events.append({"type": "slot", "player": i,
                                            "slot": inv.selected})
            attacking = bool(c.get("attack", False))
            # shield: needs owned shield; sword attack takes precedence
            if attacking or p.eating:
                p.shield_up = False
                p.shield_timer = 0
            elif bool(c.get("block", False)) and inv.available("shield"):
                if not p.shield_up:
                    if p.shield_timer <= 0:
                        p.shield_timer = 2  # raise time
                    else:
                        p.shield_timer -= 1
                        if p.shield_timer <= 0:
                            p.shield_up = True
            else:
                p.shield_up = False
                p.shield_timer = 0
            # item use
            if bool(c.get("use_item", False)) and not attacking:
                ev = USE.use_selected(self, i)
                if ev is not None:
                    self.utility_events.append(ev)
                    self.items_used[i] += 1
            ev = USE.tick_eating(self, i)
            if ev is not None:
                self.utility_events.append(ev)
                if ev["type"] == "use_complete":
                    self.items_used[i] += 1
            inv.tick()
            FX.tick_effects(p)
            # hunger drain + lava
            if self.tick % 400 == 0 and self.tick > 0:
                p.hunger = max(0.0, p.hunger - 1.0)
            fl = self.field.fluid_at(float(p.pos[0]), float(p.pos[2]))
            if fl == "lava" and not FX.has_effect(p, "fire_resistance"):
                if self.tick % 10 == 0:
                    dmg = 2.0
                    absorbed = min(p.absorption_hp, dmg)
                    p.absorption_hp -= absorbed
                    p.health = max(0.0, p.health - (dmg - absorbed))
                    if p.health <= 0.0:
                        p.alive = False
                    self.utility_events.append({"type": "lava_burn",
                                                "player": i})

    def _step_projectiles(self) -> None:
        dt = self.phys.dt
        for pr in self.projectiles:
            ev = pr.step(self, dt)
            if ev is not None:
                self.utility_events.append(ev)
                if ev["type"] == "arrow_hit":
                    atk = ev["owner"]
                    self.damage_dealt[atk] += ev["damage"]
                    self.hits[atk] += 1
                    self.last_hit_tick[atk] = self.tick
        self.projectiles = [pr for pr in self.projectiles if not pr.dead]

    def _valid(self) -> bool:
        for p in self.players:
            if not np.all(np.isfinite(p.pos)) or not np.all(np.isfinite(p.vel)):
                return False
            if not np.isfinite(p.health):
                return False
        return True

    def _event_summary(self) -> dict:
        return {"tick": self.tick, "events": [], "done": True,
                "winner": self.winner, "invalid": self.invalid}

    # -- helpers -----------------------------------------------------------
    def distance(self) -> float:
        d = self.players[0].pos - self.players[1].pos
        return float(np.linalg.norm(d))

    def state_snapshot(self) -> dict:
        return {"tick": self.tick, "done": self.done, "winner": self.winner,
                "p0": self.players[0].snapshot(), "p1": self.players[1].snapshot(),
                "damage_dealt": list(self.damage_dealt), "hits": list(self.hits),
                "combo": list(self.combo)}
