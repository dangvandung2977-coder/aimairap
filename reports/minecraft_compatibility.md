# Training → Minecraft compatibility audit

Date: 2026-09-23. Status: **bridge CONNECTED + INFERENCE_WORKING (Python side
verified); CONTROL/COMBAT/POLICY require a live Minecraft (see §20).**

## Checkpoint

- file: `checkpoints/phase4_stage3/3a2/final.zip` (untouched, read-only load)
- SHA256: `660d524dab2010fc390d2a82be52a7735f2f307ad3291c835f1ac23e7e8d5e3e` — MATCH
- model: PPO / ActorCriticPolicy (MlpPolicy [256, 256], FlattenExtractor), SB3 2.9.0

## Observation (73, obs v4)

- dimension: 73 == `OBS_DIMS[4]` (`rl/environment/obs_schema.py:11`); index
  table: SELF 0–12, ENEMY 13–25, COMBAT 26–30, WORLD 31–38, V2 39–43, V3 44–46,
  V4 47–72 — replicated 1:1 in `ObservationEncoder.java`.
- preprocessing: **none** (no VecNormalize in repo; schema `_clip` to
  [-5,5] + float32). Inference applies float32 cast + finite check only.
- normalizers reused verbatim: /size, /10, /12, /20, /100, /1200, /16, /8,
  /90, sin/cos yaw, bearing `atan2(dz,dx) − yaw`, 4-entry foe position
  history, `min(t,100)` counters, wall distances `(x−lo, hi−x, z−lo, hi−z)`,
  4-cell obstacle sensor, first-projectile relative offset.

## Action (MultiDiscrete[5,2,2,2,2,2,9,3,3])

- mapping: exact copy of `to_controller()` (`rl/environment/action_map.py:29`):
  move 0 stop/1 fwd/2 back/3 left/4 right; sprint; jump; attack (beats block);
  block (shield, 2-tick raise); use (item pipeline); slot 0–8 =
  sword/shield/bread/golden_apple/ender_pearl/cobweb/block/strength_potion/bow
  (`defs.py:76`); yaw ±15°; pitch ±10°.
- cadence: policy queried every **4 server ticks** (= training `frame_skip`);
  yaw/pitch deltas applied once per decision (trained aim dynamics preserved).

## Bridge

- protocol v1, JSON-newline, 127.0.0.1:25575 (configurable, localhost-only).
- measured (this machine): inference **0.84 ms**, well under the 5 ms target;
  bridge overhead unmeasured until live Minecraft (target <5 ms).
- safe fallback: any timeout/malformed packet → neutral controls, camera held,
  auto-reconnect; server never crashes (mod tick wrapped in try/catch).

## Minecraft side

- NeoForge 21.1.x / MC 1.21.1 / Java 21; bot + foe are server-side FakePlayers
  (real damage/knockback/death/shield/food/bow/pearl mechanics).
- supported: melee, shield, bread/gapple, pearl, bow select/draw/release,
  cobweb/block/strength slots (selectable; placement via MC mechanics later).
- unsupported by policy (slots absent): crossbow, water/lava buckets, extra
  potions — not added, per spec.

## KNOWN sim → real differences (not hidden)

1. **Yaw frame**: sim yaw 0 = +X; MC yaw 0 = +Z ⇒ `simYaw = mcYaw + 90`,
   `simPitch = −mcXRot`, deltas transfer 1:1. Verified arithmetically, not
   yet in-game.
2. **Attack cooldown**: sim counts exact ticks (/12); mod approximates from
   `getAttackStrengthScale` (`(1−strength)×20`). Same 0–20 range, ±1 tick jitter.
3. **Eating progress**: sim exposes exact ticks-left (/32); mod reports ~16
   while using. Interrupts-on-damage re-implemented mod-side (MC lacks it).
4. **Gapple cooldown**: sim 100-tick ItemCooldown; MC has none → reported 0
   (policy rarely conditions on it; re-validate in-game).
5. **Pearl ballistics**: sim pearl speed 18/gravity 12/teleport; MC pearl is a
   real thrown entity with MC physics — cutoff timing must be re-validated,
   NOT assumed transferred.
6. **Bow**: sim fires instantly on use; MC needs a real draw (~20 ticks). The
   decoder holds/releases the draw; aim/timing equivalence is degraded until
   measured. (Bow was `NOT_PROVEN` in training either — gate evidence agrees.)
7. **Movement**: sim wish-accel physics vs MC arcade velocity control;
   gravity/collision/knockback remain MC's. Speeds matched (4.3/5.6).
8. **Combo/damage/attempt counters**: tracked mod-side with sim-identical
   update rules (60-tick combo window, `min(t,100)`), seeded from live events.
9. **Obstacle sensor**: pillar-rect test against arena config instead of
   `is_solid_at`; floor/walls excluded like the sim excludes non-pillars.

## §20 checklist

- TEST A–E: **PASS** (Python side: server socket, model load, 73-dim obs in,
  legal 9-dim action out — `tests/test_deployment.py`, 10/10).
- TEST F–K: **NOT RUN** — require live Minecraft 1.21.1 + `./gradlew build`
  (no MC jars in this environment; Java syntax verified with javac, 0 syntax
  errors; API-level compile unverified).
- Verdict: CONNECTED ✔ / INFERENCE_WORKING ✔ / CONTROL_WORKING ⏳ /
  COMBAT_WORKING ⏳ / POLICY_BEHAVIOR_VALIDATED ⏳.

## Next step

`./gradlew build` in `minecraft-mod/`, drop the jar in a NeoForge 1.21.1
server `mods/`, run TEST A–K in order, then `/pvpbot dump` + golden replay
before any claim about real-Minecraft behavior.
