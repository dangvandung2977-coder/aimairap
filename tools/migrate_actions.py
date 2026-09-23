"""Migrate tests/tools to 9-branch v2 actions + v4 default obs (73).

- 6-element action literals gain [block=0, use=0, slot=0] inserted before yaw/pitch.
- (47,) /(4, 47) -> (73,) /(4, 73); OBS_DIMS[3] default asserts -> OBS_DIMS[4].
- rng.integers(5..) action samplers -> action_map.random_controller(rng).
Run: python tools/migrate_actions.py
"""
import re

FILES = ["tests/test_obs.py", "tests/test_smoke_env.py", "tests/test_phase2.py",
         "tests/test_actions.py", "tests/test_determinism2.py",
         "tests/test_reward.py", "tools/smoke_random.py", "tools/smoke_ppo.py",
         "tools/eval.py", "tools/record_demo.py", "tools/baselines.py",
         "visualization/debug_view.py"]

SIX_ACT = re.compile(
    r"np\.array\(\[rng\.integers\(5\), rng\.integers\(2\), rng\.integers\(2\),\s*"
    r"rng\.integers\(2\), rng\.integers\(3\), rng\.integers\(3\)\]\)")
SIX_ACT2 = re.compile(
    r"\[\s*rng\.integers\(5\), rng\.integers\(2\), rng\.integers\(2\),\s*"
    r"rng\.integers\(2\), rng\.integers\(3\), rng\.integers\(3\)\s*\]")


def migrate(path: str) -> None:
    s = open(path).read()
    s = SIX_ACT.sub("action_map.random_controller(rng)", s)
    s = SIX_ACT2.sub("action_map.random_controller(rng).tolist()", s)
    # literal 6-acts [a,b,c,d,e,f] used in step() calls: insert 0,0,0 before yaw
    s = re.sub(r"np\.array\(\[(\d), (\d), (\d), (\d), (\d), (\d)\]\)",
               r"np.array([\1, \2, \3, \4, 0, 0, 0, \5, \6])", s)
    s = s.replace("(47,)", "(73,)").replace("(4, 47)", "(4, 73)")
    s = s.replace("OBS_DIMS[3],) == (47,)", "OBS_DIMS[4],) == (73,)")
    s = s.replace('assert info["obs_version"] == 3',
                  'assert info["obs_version"] == 4')
    s = s.replace("assert env.observation_space.shape == (47,)",
                  "assert env.observation_space.shape == (73,)")
    s = s.replace("list(env.action_space.nvec) == [5, 2, 2, 2, 3, 3]",
                  "list(env.action_space.nvec) == [5, 2, 2, 2, 2, 2, 9, 3, 3]")
    if "action_map" not in s and "random_controller" in s:
        s = s.replace("from rl.environment.gym_env import",
                      "from rl.environment import action_map\nfrom rl.environment.gym_env import")
    open(path, "w").write(s)
    print("migrated", path)


def main() -> None:
    for path in FILES:
        try:
            migrate(path)
        except FileNotFoundError:
            print("skip", path)


if __name__ == "__main__":
    main()
