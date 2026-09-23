"""Counterfactual gate: fight-first vs eat-first under regen (scripted)."""
from sandbox.simulation.sim import PvPSim, idle_ctrl
from rl.opponents.scripted import MeleeOpponent
from rl.opponents.sustain import RegenB


def _foe(seed):
    if not hasattr(_foe, "f"):
        _foe.f = {}
    if seed not in _foe.f:
        _foe.f[seed] = RegenB(seed=seed)
    return _foe.f[seed]


def fight_first(seed, regen, hp0):
    sim = PvPSim(seed=seed)
    sim.foe_regen_rate, sim.foe_regen_delay = regen
    sim.players[0].health = hp0
    drv = MeleeOpponent()
    drv.reset(seed)
    while not sim.done:
        sim.step(drv.act(sim, 0), _foe(seed).act(sim, 1))
    return sim.winner == 0


def eat_first(seed, regen, hp0):
    sim = PvPSim(seed=seed)
    sim.foe_regen_rate, sim.foe_regen_delay = regen
    sim.players[0].health = hp0
    drv = MeleeOpponent()
    drv.reset(seed)
    phase = 0
    while not sim.done:
        c = dict(idle_ctrl())
        inv = sim.inventories[0]
        me = sim.players[0]
        if phase == 0:  # disengage + eat gapple
            if inv.selected != 3:
                c["select_slot"] = 3
            else:
                c["use_item"] = True
            c["move_z"] = -1.0
            if me.eating:
                phase = 1
        elif phase == 1:
            c["move_z"] = -1.0
            if not me.eating:
                phase = 2
        else:
            c = drv.act(sim, 0)
        sim.step(c, _foe(seed).act(sim, 1))
    return sim.winner == 0


def main() -> None:
    for hp0 in (10.0, 12.0, 14.0):
        for regen, tag in [((0.0, 0), "no-regen"), ((0.05, 60), "regenB")]:
            _foe.f = {}
            f = sum(fight_first(s, regen, hp0) for s in range(5000, 5010))
            _foe.f = {}
            e = sum(eat_first(s, regen, hp0) for s in range(5000, 5010))
            print(f"hp={hp0} {tag}: fight-first {f}/10, eat-first {e}/10")


if __name__ == "__main__":
    main()
