"""Generate Phase-3 screening configs from phase275 base. One variable each.

  python -m tools.mk_mech
Writes configs/mech/{base,A_crit10,B_air10,C_arcV45,D_land4}.yaml
All share: seed 51, obs v3, same reward, same grounded curriculum, 400k steps.
"""
import copy

import yaml

BASE = "configs/phase275_melee.yaml"
OUT = "configs/mech"

VARIANTS = {
    # name: (mechanics label, combat overrides, physics overrides)
    "base": ("baseline (crit1.5/air2.5/arcV75/land0)", {}, {}),
    "A_crit10": ("crit_mult 1.5 -> 1.0", {"crit_mult": 1.0}, {}),
    "B_air10": ("air_accel 2.5 -> 1.0", {}, {"air_accel": 1.0}),
    "C_arcV45": ("vertical arc 75 -> 45 (horizontal stays 75)",
                 {"attack_arc_vertical_deg": 45.0}, {}),
    "D_land4": ("landing attack lockout 0 -> 4 ticks", {"landing_attack_delay_ticks": 4}, {}),
}


def main() -> None:
    import os
    os.makedirs(OUT, exist_ok=True)
    with open(BASE) as f:
        base = yaml.safe_load(f)
    for name, (label, combat, physics) in VARIANTS.items():
        cfg = copy.deepcopy(base)
        cfg["run"]["name"] = f"mech_{name}"
        cfg["run"]["seed"] = 51
        cfg["training"]["timesteps"] = 400000
        cfg["eval"]["freq"] = 100000
        cfg["checkpoint"]["dir"] = f"checkpoints/mech_{name}"
        cfg["checkpoint"]["freq"] = 200000
        cfg["logging"]["tensorboard"] = f"logs/mech_{name}"
        cfg["combat"] = combat
        cfg["physics"] = physics
        cfg["mechanics_label"] = label
        path = f"{OUT}/{name}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)
        print(path, "::", label)


if __name__ == "__main__":
    main()
