"""Stage-2 sustain configs 2A-2E. Reward byte-identical. D mechanics. v4/9-branch.
From micro champ (phase4_1h3/micro/final). Gates between stages.
"""
import yaml

SW = ["sword", "shield"]
SUSTAIN = ["sword", "shield", "bread", "golden_apple"]
BASE_LOADOUT = {
    "sword": 1, "shield": 1, "bread": 5, "golden_apple": 2,
    "ender_pearl": 4, "cobweb": 4, "block": 16, "strength_potion": 1,
    "bow": 1, "arrow": 16,
}
LEAN_LOADOUT = dict(BASE_LOADOUT)
LEAN_LOADOUT.update({"bread": 2, "golden_apple": 1})


def base(name: str, ckpt: str, log: str, steps: int) -> dict:
    cfg = yaml.safe_load(open("configs/phase4_stage01.yaml"))
    cfg["run"]["name"] = name
    cfg["training"]["timesteps"] = steps
    cfg["checkpoint"]["dir"] = ckpt
    cfg["checkpoint"]["freq"] = 100000
    cfg["logging"]["tensorboard"] = log
    cfg["env"]["obs_version"] = 4
    return cfg


def write(path: str, cfg: dict) -> None:
    yaml.safe_dump(cfg, open(path, "w"), default_flow_style=False)
    print("wrote", path)


def main() -> None:
    # 2A: bread only, low-pressure melee/chaser, full HP
    cfg = base("stage2a_food", "checkpoints/phase4_stage2/2a",
               "logs/phase4_stage2_2a", 250000)
    cfg["env"]["loadout"] = dict(BASE_LOADOUT)
    cfg["env"]["enabled_items"] = SW + ["bread"]
    cfg["env"]["opponent_pool"] = [
        {"until": None, "enabled": SW + ["bread"],
         "pool": [
             {"opponent": "melee", "kwargs": {"seed": 501},
              "enabled": SW + ["bread"]},
             {"opponent": "chaser", "kwargs": {"seed": 502},
              "enabled": SW + ["bread"]}]},
    ]
    write("configs/stage2a.yaml", cfg)

    # 2B: +gapple, low-HP starts (safe/unsafe mix via HP range + foe mix)
    cfg = base("stage2b_gapple", "checkpoints/phase4_stage2/2b",
               "logs/phase4_stage2_2b", 250000)
    cfg["env"]["loadout"] = dict(BASE_LOADOUT)
    cfg["env"]["enabled_items"] = SUSTAIN
    cfg["env"]["agent_start_hp"] = [6.0, 20.0]
    cfg["env"]["opponent_pool"] = [
        {"until": None, "enabled": SUSTAIN,
         "pool": [
             {"opponent": "melee", "kwargs": {"seed": 511},
              "enabled": SUSTAIN},
             {"opponent": "chaser", "kwargs": {"seed": 512},
              "enabled": SUSTAIN},
             {"opponent": "gappleuser", "kwargs": {"seed": 513},
              "enabled": SUSTAIN}]},
    ]
    write("configs/stage2b.yaml", cfg)

    # 2C: pressure/punish eaters (same unlocks)
    cfg = base("stage2c_pressure", "checkpoints/phase4_stage2/2c",
               "logs/phase4_stage2_2c", 300000)
    cfg["env"]["loadout"] = dict(BASE_LOADOUT)
    cfg["env"]["enabled_items"] = SUSTAIN
    cfg["env"]["agent_start_hp"] = [6.0, 20.0]
    cfg["env"]["opponent_pool"] = [
        {"until": None, "enabled": SUSTAIN,
         "pool": [
             {"opponent": "pressureeater", "kwargs": {"seed": 521},
              "enabled": SUSTAIN},
             {"opponent": "punisheater", "kwargs": {"seed": 522},
              "enabled": SUSTAIN},
             {"opponent": "mixedeater", "kwargs": {"seed": 523},
              "enabled": SUSTAIN},
             {"opponent": "aggressive", "kwargs": {"seed": 524},
              "enabled": SUSTAIN}]},
    ]
    write("configs/stage2c.yaml", cfg)

    # 2D: finite resources (lean loadout, frequent low HP)
    cfg = base("stage2d_resource", "checkpoints/phase4_stage2/2d",
               "logs/phase4_stage2_2d", 250000)
    cfg["env"]["loadout"] = dict(LEAN_LOADOUT)
    cfg["env"]["enabled_items"] = SUSTAIN
    cfg["env"]["agent_start_hp"] = [5.0, 14.0]
    cfg["env"]["opponent_pool"] = [
        {"until": None, "enabled": SUSTAIN, "loadout": dict(LEAN_LOADOUT),
         "pool": [
             {"opponent": "melee", "kwargs": {"seed": 531},
              "enabled": SUSTAIN},
             {"opponent": "punisheater", "kwargs": {"seed": 532},
              "enabled": SUSTAIN},
             {"opponent": "aggressive", "kwargs": {"seed": 533},
              "enabled": SUSTAIN}]},
    ]
    write("configs/stage2d.yaml", cfg)

    # 2E: consolidated (90% stage-2 foes, 10% rehearsal via sample weights)
    cfg = base("stage2e_consolidated", "checkpoints/phase4_stage2/2e",
               "logs/phase4_stage2_2e", 400000)
    cfg["env"]["loadout"] = dict(BASE_LOADOUT)
    cfg["env"]["enabled_items"] = SUSTAIN
    cfg["env"]["agent_start_hp"] = [6.0, 20.0]
    cfg["env"]["opponent_pool"] = [
        {"until": None, "enabled": SUSTAIN,
         "sample_pool": [
             {"opponent": "punisheater", "kwargs": {}, "weight": 3.0},
             {"opponent": "mixedeater", "kwargs": {}, "weight": 3.0},
             {"opponent": "pressureeater", "kwargs": {}, "weight": 2.0},
             {"opponent": "gappleuser", "kwargs": {}, "weight": 1.0},
             {"opponent": "shieldmixed", "kwargs": {}, "weight": 0.5},
             {"opponent": "circlerL", "kwargs": {}, "weight": 0.25},
             {"opponent": "circlerR", "kwargs": {}, "weight": 0.25}]},
    ]
    write("configs/stage2e.yaml", cfg)


if __name__ == "__main__":
    main()
