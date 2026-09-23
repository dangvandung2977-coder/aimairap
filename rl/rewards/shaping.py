"""Dense reward, config-driven. Components logged separately."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    w_damage_dealt: float = 0.20
    w_damage_taken: float = -0.12
    kill: float = 10.0
    death: float = -10.0
    win: float = 5.0     # bonus on top of kill when winner == agent
    loss: float = -5.0
    engage_bonus: float = 0.01   # per decision while dist < engage_range
    engage_range: float = 6.0
    face_bonus: float = 0.005    # per decision while roughly facing foe
    step_penalty: float = -0.001
    timeout_penalty: float = -1.0


COMPONENTS = ("damage_dealt", "damage_taken", "kill", "death", "win_loss",
              "engage", "face", "step", "timeout")


class RewardBook:
    """Tracks last-paid damage so per-decision deltas are exact."""

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.paid_dealt = 0.0
        self.paid_taken = 0.0
        self.totals = {k: 0.0 for k in COMPONENTS}

    def reset(self):
        self.paid_dealt = 0.0
        self.paid_taken = 0.0
        self.totals = {k: 0.0 for k in COMPONENTS}

    def step(self, sim, agent: int, done: bool, facing: bool) -> tuple[float, dict]:
        c = self.cfg
        dealt_now = sim.damage_dealt[agent]
        taken_now = sim.damage_dealt[1 - agent]
        d_dealt = dealt_now - self.paid_dealt
        d_taken = taken_now - self.paid_taken
        self.paid_dealt, self.paid_taken = dealt_now, taken_now
        comp = {k: 0.0 for k in COMPONENTS}
        comp["damage_dealt"] = c.w_damage_dealt * d_dealt
        comp["damage_taken"] = c.w_damage_taken * d_taken
        comp["step"] = c.step_penalty
        if sim.distance() < c.engage_range:
            comp["engage"] = c.engage_bonus
        if facing:
            comp["face"] = c.face_bonus
        if done:
            if sim.winner == agent:
                comp["kill"] = c.kill
                comp["win_loss"] = c.win
            elif sim.winner == 1 - agent:
                comp["death"] = c.death
                comp["win_loss"] = c.loss
            elif sim.tick >= sim.max_ticks:
                comp["timeout"] = c.timeout_penalty
        total = sum(comp.values())
        for k, v in comp.items():
            self.totals[k] += v
        return float(total), comp
