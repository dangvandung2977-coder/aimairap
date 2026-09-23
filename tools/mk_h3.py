"""H-3 matrix configs. All derive from phase4_1h2.yaml; only pool/steps/dirs
change. Budgets matched (200k per phase). Parent checkpoints recorded."""
import yaml

SW = ["sword", "shield"]
SHIELD = [
    ("shieldmixed", 301), ("shieldholder", 302), ("turtleB", 303),
    ("shieldbaiter", 304), ("shieldpressure", 305),
]
CIRCLER = [
    ("circlerL", 311), ("circlerR", 312), ("speeder", 313),
    ("switcher", 314),
]


def pool(entries):
    return [{"opponent": n, "kwargs": {"seed": s}, "enabled": SW}
            for n, s in entries]


def write(name, stages, steps, ckpt, log):
    cfg = yaml.safe_load(open("configs/phase4_1h2.yaml"))
    cfg["run"]["name"] = name
    cfg["training"]["timesteps"] = steps
    cfg["env"]["opponent_pool"] = [
        {"until": until, "pool": pool(entries)}
        for until, entries in stages
    ]
    cfg["checkpoint"]["dir"] = ckpt
    cfg["checkpoint"]["freq"] = 100000
    cfg["checkpoint"]["keep"] = 6
    cfg["logging"]["tensorboard"] = log
    path = f"configs/{name}.yaml"
    yaml.safe_dump(cfg, open(path, "w"), default_flow_style=False)
    print("wrote", path)


def main() -> None:
    # Run A: shield-dominant then circler-dominant (two separate runs)
    write("h3_Ashield", [(None, SHIELD)], 200000,
          "checkpoints/phase4_1h3/Ashield", "logs/phase4_1h3_Ashield")
    write("h3_Acircler", [(None, CIRCLER)], 200000,
          "checkpoints/phase4_1h3/Acircler", "logs/phase4_1h3_Acircler")
    # Run B (reverse): circler-dominant then shield-dominant
    write("h3_Bcircler", [(None, CIRCLER)], 200000,
          "checkpoints/phase4_1h3/Bcircler", "logs/phase4_1h3_Bcircler")
    write("h3_Bshield", [(None, SHIELD)], 200000,
          "checkpoints/phase4_1h3/Bshield", "logs/phase4_1h3_Bshield")


if __name__ == "__main__":
    main()
