"""Stage-3 configs. 3A first; later stages gated on 3A results."""
import yaml

PEARL_UNLOCK = ["sword", "shield", "bread", "golden_apple", "ender_pearl"]
PEARL_LOADOUT = {
    "sword": 1, "shield": 1, "bread": 2, "golden_apple": 1,
    "ender_pearl": 4, "cobweb": 0, "block": 0, "strength_potion": 0,
    "bow": 1, "arrow": 0,
}


def base(name: str, ckpt: str, log: str, steps: int) -> dict:
    cfg = yaml.safe_load(open("configs/phase4_stage01.yaml"))
    cfg["run"]["name"] = name
    cfg["training"]["timesteps"] = steps
    cfg["checkpoint"]["dir"] = ckpt
    cfg["checkpoint"]["freq"] = 100000
    cfg["logging"]["tensorboard"] = log
    cfg["env"]["obs_version"] = 4
    cfg["env"]["loadout"] = dict(PEARL_LOADOUT)
    cfg["env"]["enabled_items"] = list(PEARL_UNLOCK)
    # 3A kite-cutoff arena: big + divider wall (scripted gate: pearl 9/10
    # wins vs 10/10 draws without; open arena is caught on foot instead)
    cfg["env"]["arena_size"] = 32
    cfg["env"]["obstacles"] = [[15.5, 4.0, 16.5, 28.0, 3.0]]
    return cfg


def main() -> None:
    cfg = base("stage3a_pearl_basic", "checkpoints/phase4_stage3/3a",
               "logs/phase4_stage3_3a", 300000)
    cfg["env"]["opponent_pool"] = [
        {"until": None, "enabled": list(PEARL_UNLOCK),
         "sample_pool": [
             {"opponent": "farfighter", "kwargs": {}, "weight": 6.0},
             {"opponent": "melee", "kwargs": {}, "weight": 2.0},
             {"opponent": "shieldmixed", "kwargs": {}, "weight": 0.5},
             {"opponent": "circlerL", "kwargs": {}, "weight": 0.25},
             {"opponent": "circlerR", "kwargs": {}, "weight": 0.25}]},
    ]
    yaml.safe_dump(cfg, open("configs/stage3a.yaml", "w"),
                   default_flow_style=False)
    print("wrote configs/stage3a.yaml")


if __name__ == "__main__":
    main()
