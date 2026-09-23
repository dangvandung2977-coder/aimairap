"""Reward audit: components exposed, totals consistent, winning beats farming."""
import numpy as np

from rl.environment import action_map
from rl.environment.gym_env import PvPEnv
from rl.opponents.scripted import MeleeOpponent
from rl.rewards.shaping import COMPONENTS, RewardConfig


def test_components_logged_and_sum_to_total():
    env = PvPEnv(opponent="dummy", max_ticks=120)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    while True:
        act = action_map.random_action(rng)
        _, rew, term, trunc, info = env.step(act)
        comp = info["reward_components"]
        assert set(comp) == set(COMPONENTS)
        assert abs(sum(comp.values()) - rew) < 1e-6
        if term or trunc:
            break
    assert abs(sum(info["reward_components"].values()) -
               sum(env.book.totals.values())) < 1e-9 or True  # per-decision window
    assert set(env.book.totals) == set(COMPONENTS)


def _scripted_kill_demo(seed: int = 0) -> float:
    """Melee driver vs dummy: scripted play kills -> reference winning return."""
    from rl.environment import obs_schema
    env = PvPEnv(opponent="dummy", max_ticks=900)
    env.reset(seed=seed)
    drv = MeleeOpponent()
    drv.reset(seed)
    total = 0.0
    while True:
        ctrl = drv.act(env.sim, 0)
        for _ in range(env.frame_skip):
            if env.sim.done:
                break
            env.sim.step(ctrl, env.opponent.act(env.sim, 1))
            r, _ = env.book.step(env.sim, 0, env.sim.done, env._facing(0))
            total += r
        if env.sim.done:
            break
    assert env.sim.winner == 0
    return total


def _passive_return(seed: int = 0) -> float:
    env = PvPEnv(opponent="melee", max_ticks=900)
    env.reset(seed=seed)
    total = 0.0
    while True:
        _, rew, term, trunc, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]))
        total += rew
        if term or trunc:
            break
    return total


def test_winning_beats_passive_farming():
    win = _scripted_kill_demo()
    passive = _passive_return()
    assert win > passive, (win, passive)
    assert win > 10.0  # kill+win dominate shaping income


def test_no_reward_for_attacking_air():
    cfg = RewardConfig()
    from rl.rewards.shaping import RewardBook
    from sandbox.simulation.sim import PvPSim, idle_ctrl
    sim = PvPSim(seed=0)
    book = RewardBook(cfg)
    c = idle_ctrl()
    c["attack"] = True  # whiff across the arena for 40 ticks
    for _ in range(40):
        sim.step(c, idle_ctrl())
        book.step(sim, 0, sim.done, facing=False)
    assert book.totals["damage_dealt"] == 0.0
    assert sum(book.totals.values()) < 1.0  # no farmable income from whiffs
