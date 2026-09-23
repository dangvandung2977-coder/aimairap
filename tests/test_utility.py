"""Phase 4 utility tests: every system + determinism. No reward involved."""
import numpy as np

from sandbox.combat.melee import try_attack
from sandbox.config import CombatConfig
from sandbox.items import effects as FX
from sandbox.items.inventory import Inventory
from sandbox.simulation.sim import PvPSim, idle_ctrl


def _ctrl(**kw):
    c = idle_ctrl()
    c.update(kw)
    return c


def _use_slot(sim, slot: int):
    """Select a slot (pay switch cost), then trigger use on a later tick."""
    sim.step(_ctrl(select_slot=slot), idle_ctrl())
    sim.step(_ctrl(), idle_ctrl())
    sim.step(_ctrl(), idle_ctrl())
    sim.step(_ctrl(use_item=True), idle_ctrl())


def test_inventory_counts_switch_consume():
    inv = Inventory({"bread": 3, "ender_pearl": 1})
    assert inv.available("bread") and inv.available("ender_pearl")
    assert not inv.available("cobweb")
    assert inv.selected_item() == "sword"
    assert inv.select(4) and inv.selected_item() == "ender_pearl"
    assert inv.switch_timer == 2
    assert inv.consume("ender_pearl") and not inv.available("ender_pearl")
    inv.enabled = {"sword"}
    assert not inv.available("bread")  # curriculum lock


def test_shield_blocks_facing_attack():
    cfg = CombatConfig()
    a, b = None, None
    from sandbox.entities.player import PlayerState
    a, b = PlayerState(), PlayerState()
    a.reset([10, 0, 10], yaw=0.0)
    b.reset([12, 0, 10], yaw=180.0)  # faces attacker
    b.shield_up = True
    ev = try_attack(a, b, cfg)
    assert ev.get("blocked") and ev["damage"] == 0.0 and b.health == 20.0
    # facing away: no block
    b2 = PlayerState()
    b2.reset([12, 0, 10], yaw=0.0)
    b2.shield_up = True
    a.attack_cooldown = 0
    ev2 = try_attack(a, b2, cfg)
    assert not ev2.get("blocked") and ev2["hit"]


def test_shield_raise_timing_in_sim():
    sim = PvPSim(seed=0)
    sim.step(_ctrl(block=True), idle_ctrl())
    assert not sim.players[0].shield_up  # raising
    sim.step(_ctrl(block=True), idle_ctrl())
    sim.step(_ctrl(block=True), idle_ctrl())
    assert sim.players[0].shield_up
    sim.step(_ctrl(attack=True), idle_ctrl())  # sword takes precedence
    assert not sim.players[0].shield_up


def test_food_eating_and_interruption():
    sim = PvPSim(seed=0)
    sim.players[0].health = 10.0
    _use_slot(sim, 2)  # bread
    assert sim.players[0].eating[0] == "bread"
    for _ in range(31):
        sim.step(_ctrl(), idle_ctrl())
    assert sim.players[0].eating == [] and sim.players[0].health == 12.0
    assert sim.inventories[0].counts["bread"] == 4
    # interruption: damage cancels, item kept
    sim2 = PvPSim(seed=0)
    _use_slot(sim2, 2)
    assert sim2.players[0].eating[0] == "bread"
    sim2.players[1].pos[:] = [12, 0, 10]
    sim2.players[1].yaw = 180.0
    for _ in range(5):
        atk = _ctrl()
        atk2 = _ctrl(attack=True)
        sim2.step(atk, atk2)
        if sim2.players[0].eating == []:
            break
    assert sim2.inventories[0].counts["bread"] == 5  # not consumed


def test_gapple_absorption_regen():
    sim = PvPSim(seed=0)
    sim.players[0].health = 10.0
    _use_slot(sim, 3)
    for _ in range(40):
        sim.step(_ctrl(), idle_ctrl())
    assert sim.players[0].absorption_hp == 8.0
    assert FX.has_effect(sim.players[0], "regeneration")
    assert sim.players[0].health > 10.0


def test_pearl_throw_teleport():
    sim = PvPSim(seed=0)
    p0 = sim.players[0].pos.copy()
    _use_slot(sim, 4)
    assert len(sim.projectiles) == 1
    for _ in range(200):
        sim.step(_ctrl(), idle_ctrl())
        if not sim.projectiles:
            break
    assert not sim.projectiles
    moved = float(np.linalg.norm(sim.players[0].pos - p0))
    assert moved > 3.0  # teleported
    assert sim.inventories[0].counts["ender_pearl"] == 3


def test_cobweb_slow():
    sim = PvPSim(seed=0)
    _use_slot(sim, 5)
    assert len(sim.field.webs) == 1
    w = sim.field.webs[0]
    sim.players[0].pos[:] = [w.x, 0, w.z]
    c = _ctrl(move_z=1.0)
    from sandbox.physics.engine import step_player
    v_free = None
    sim2 = PvPSim(seed=0)
    p = sim2.players[0]
    for _ in range(30):
        step_player(p, c, sim2.world, sim2.phys)
    v_free = float(np.linalg.norm(p.vel[[0, 2]]))
    p1 = sim.players[0]
    for _ in range(10):
        p1.pos[:] = [w.x, 0, w.z]  # pin inside web
        step_player(p1, c, sim.world, sim.phys)
    v_web = float(np.linalg.norm(p1.vel[[0, 2]]))
    assert v_web < v_free * 0.6, (v_web, v_free)


def test_block_place_collide_remove():
    sim = PvPSim(seed=0)
    me = sim.players[0]
    me.pos[:] = [10, 0, 10]
    me.yaw = 0.0
    _use_slot(sim, 6)
    assert len(sim.field.blocks) == 1
    b = sim.field.blocks[0]
    assert sim.world.is_solid_at(b.x, 0.5, b.z)
    # whiffed swing at the block breaks it
    me.pos[:] = [b.x - 1.5, 0, b.z]
    me.yaw = 0.0
    for _ in range(30):
        sim.step(_ctrl(attack=True), idle_ctrl())
        if not sim.field.blocks:
            break
    assert not sim.field.blocks


def test_lava_damage_and_fire_res():
    sim = PvPSim(seed=0)
    sim.field.add_fluid("lava", 10.0, 10.0, sim.world.size)
    sim.players[0].pos[:] = [10, 0, 10]
    hp0 = sim.players[0].health
    for _ in range(30):
        sim.step(_ctrl(), idle_ctrl())
    assert sim.players[0].health < hp0
    sim2 = PvPSim(seed=0)
    FX.apply_effect(sim2.players[0], "fire_resistance", 1200, 0)
    sim2.field.add_fluid("lava", 10.0, 10.0, sim2.world.size)
    sim2.players[0].pos[:] = [10, 0, 10]
    for _ in range(30):
        sim2.step(_ctrl(), idle_ctrl())
    assert sim2.players[0].health == 20.0


def test_potion_speed_strength():
    sim = PvPSim(seed=0)
    sim.inventories[0].counts["strength_potion"] = 1
    sim.players[0].health = 20.0
    _use_slot(sim, 7)
    for _ in range(40):
        sim.step(_ctrl(), idle_ctrl())
    assert FX.has_effect(sim.players[0], "strength")
    assert FX.damage_mult(sim.players[0]) == 1.3


def test_bow_shot_consumes_arrow():
    sim = PvPSim(seed=0)
    sim.players[1].pos[:] = [13, 0, 10]
    sim.players[0].yaw = 0.0
    n0 = sim.inventories[0].counts["arrow"]
    _use_slot(sim, 8)
    assert sim.inventories[0].counts["arrow"] == n0 - 1
    for _ in range(100):
        sim.step(_ctrl(), idle_ctrl())
        if sim.damage_dealt[0] > 0:
            break
    assert sim.damage_dealt[0] == 6.0


def test_utility_determinism():
    def run():
        sim = PvPSim(seed=9)
        seq = [dict(block=True), dict(use_item=True, select_slot=4)]
        for t in range(120):
            c0 = _ctrl(**(seq[t % 2] if t % 10 == 0 else {}))
            sim.step(c0, idle_ctrl())
        p = sim.players[0]
        return (p.pos.copy(), p.health, p.absorption_hp,
                sim.inventories[0].counts.get("ender_pearl"),
                len(sim.field.webs), sim.tick)
    a, b = run(), run()
    assert a[0].tolist() == b[0].tolist() and a[1:] == b[1:]


def test_invalid_use_no_reward_effect():
    sim = PvPSim(seed=0)
    sim.inventories[0].counts["ender_pearl"] = 0
    hp = sim.players[0].health
    sim.step(_ctrl(select_slot=4), idle_ctrl())
    sim.step(_ctrl(), idle_ctrl())
    sim.step(_ctrl(), idle_ctrl())
    for _ in range(10):
        sim.step(_ctrl(use_item=True), idle_ctrl())
    assert sim.players[0].health == hp and not sim.projectiles
