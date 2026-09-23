# Minecraft-like PvP RL Sandbox — Phase 1.5 baseline
Deterministic, headless, Gymnasium-compatible 1v1 melee simulator + PPO vs scripted melee.

## Commands (VPS-ready, all headless except viewer/tensorboard)
```bash
pip install -r requirements.txt
pytest -q                                                        # 1. unit tests
pytest tests/test_determinism2.py -q                             # 2. determinism
pytest tests/test_smoke_env.py -q                                # 3. env smoke
python -m tools.baselines --episodes 10                          # 4+5. random+scripted baselines
python -m tools.smoke_ppo                                        # 6. PPO smoke
python -m tools.eval --checkpoint <ckpt> --opponent melee        # 7. evaluate
python train.py --config configs/baseline_melee.yaml             # 8. long training
python train.py --resume checkpoints/baseline/latest.zip         # 9. resume
python -m replay.viewer --replay replays/eval/<f>.npz            # 10. replay viewer
tensorboard --logdir logs/baseline                               # 11. metrics
python -m tools.bench                                            # VPS sizing
python -m tools.audit_behavior --replay <f>.npz                  # anti-hack check
```

## Verified results
Phase 1 (vs dummy, 150k): 100% wins, 22.5 dmg, 14 decisions/kill, agent untouched.
Phase 1.5 (PPO vs scripted melee, 800k + resume → 1.8M after a server restart):
- A random: 0% wins / 3.75 dmg. B scripted mirror: 0% (P0 side) / 96.7% hit rate.
- C PPO (`checkpoints/baseline/best_model.zip`): **100% wins / 22.5 dmg dealt /
  14.25 taken / 55% hit rate** over 10 seeded episodes (`logs/baselines.json`).
- At the 1M crossing: win_rate 1.0, reward 18.36 — improvement is real, not a crash-free process.
- 43 periodic evals in `logs/baseline/eval_combat.jsonl` + TensorBoard
  (eval_combat/*, eval_reward_comp/*) + per-eval replay; champ audit PASS.
- Crash-recovery proven: restart at 800k → `--resume latest.zip` continued to 1.8M.

## Phase 2 (pure-melee upgrade, 2M steps, obs v2, curriculum)
- Baseline = `checkpoints/baseline/best_model.zip` (obs v1, 1.8M vs melee).
- Phase 2 = `checkpoints/phase2/best_model.zip` (obs v2 44-dim, curriculum
  melee/chaser → +strafer/aggressive → randomized pool, 2M steps).
- Suite: 5 profiles × 30 held-out seeds (5000+), per-opponent + per-episode
  detail in `logs/suite_baseline.json` / `logs/suite_phase2.json`.

| profile    | win base→p2 | dealt | taken | hit | dist | hp left |
| chaser     | 1.00→1.00 | 21.1→21.2 | 13.2→13.3 | .60→.55 | 3.9→4.1 | 6.8→6.7 |
| strafer    | .97→.97 | 21.1→21.5 | 14.8→13.6 | .52→.46 | 4.2→4.5 | 5.2→6.4 |
| aggressive | .83→.90 | 19.8→20.5 | 15.4→14.3 | .54→.51 | 4.0→4.3 | 4.8→5.8 |
| defensive  | 1.00→1.00 | 21.6→21.2 | 7.8→7.4 | .55→.45 | 4.4→4.7 | 12.2→12.6 |
| random     | 1.00→.93 | 21.3→21.1 | 13.8→13.2 | .55→.50 | 4.0→4.2 | 6.2→6.8 |

- Honest read: small robustness gain (±2/30 eps is noise; CIs overlap),
  slightly better survival, slightly worse accuracy. The champ still
  bunny-hop crit-spams with strafe_fraction 0.0 — jump-crit is OPTIMAL vs
  these bots (3-hit kills), so no strafing emerged. Not a hack (audit PASS,
  takes real damage, wins real fights); the fix needs punishing opponents,
  not reward tweaks.
- Throughput: clean bench `logs/bench_phase2.json` (19.4k t/s @4envs;
  phase-1.5 numbers were measured under load, not comparable). Training held
  ~680 fps, n_envs=2 kept for 2-vCPU. 41/41 tests pass.

## Phase 2.5 (counterplay, 1.5M steps, obs v3, reward UNCHANGED)
- Champ = `checkpoints/phase25/best_model.zip` (v3 47-dim, 6-stage curriculum
  melee/chaser → backoff/strafepress → land/misspunish → mixed → random pool).
- Suite 10 profiles × 30 held-out seeds: `logs/suite_phase25.json` (+per-ep detail).

| profile    | win p2→p25 | dealt | taken | hit | dist |
| chaser     | 1.00→.97 | 21.2→20.5 | 13.3→13.8 | .55→.62 | 4.1→3.0 |
| strafer    | .97→.87 | 21.5→20.1 | 13.6→14.7 | .46→.58 | 4.5→3.6 |
| aggressive | .90→.80 | 20.5→19.5 | 14.3→15.7 | .51→.60 | 4.3→3.1 |
| defensive  | 1.00→1.00 | 21.2→21.1 | 7.4→7.1 | .45→.63 | 4.7→3.6 |
| random     | .93→.93 | 21.1→20.3 | 13.2→12.9 | .50→.61 | 4.2→3.3 |
| backoff    | .87→1.00 | 20.5→20.8 | 13.7→13.5 | .34→.62 | 4.7→3.4 |
| landpunish | .00→1.00 | 0.0→22.2 | 0.0→4.5 | .00→.63 | 3.7→3.9 |
| misspunish | 1.00→1.00 | 21.3→21.6 | 13.7→13.2 | .46→.62 | 4.7→3.5 |
| strafepress| .83→.80 | 20.2→19.2 | 12.5→15.2 | .31→.52 | 4.2→3.7 |
| mixed      | 1.00→.87 | 21.9→20.0 | 13.7→14.7 | .44→.58 | 4.4→3.5 |

- Behavior (`logs/phase25_behavior.json` vs `logs/phase25_problem.json`):
  jump ~0.95 (UNCHANGED), strafe 0.0→0.16–0.39 (emerged, varies by foe),
  hit rate up everywhere, mean dist down ~1 block (tighter spacing).
- Counterplay verdict: landpunish solved (0→100%, 4.5 taken), backoff fixed
  (.87→1.00). Cost: lateral-foe wins dipped (strafer/aggressive/strafepress
  −7–10pp) — tighter, more accurate fighter that trades worse vs dodgers.
- 45/45 tests pass. Bench idle: 12.5k(2)/20.5k(4)/24.4k(8) t/s — v3+counters
  cost nothing measurable; n_envs=2 kept.

## Phase 2.75 (grounded viability, 1.5M steps, obs v3, reward UNCHANGED)
- Champ = `checkpoints/phase275/best_model.zip` (v3, 7-stage grounded
  curriculum). Old checkpoints untouched.
- Suite 14 profiles × 30 held-out seeds: `logs/suite_phase275.json`.

| profile    | win p25→p275 | dealt | taken | hit |
| chaser     | .97→.90 | 20.5→20.2 | 13.8→15.0 | .62→.60 |
| strafer    | .87→1.00 | 20.1→21.6 | 14.7→13.9 | .58→.62 |
| aggressive | .80→.73 | 19.5→19.2 | 15.7→16.8 | .60→.56 |
| defensive  | 1.00→1.00 | 21.1→21.7 | 7.1→7.7 | .63→.60 |
| random     | .93→.90 | 20.3→20.8 | 12.9→14.8 | .61→.59 |
| backoff    | 1.00→.93 | 20.8→21.4 | 13.5→14.8 | .62→.56 |
| landpunish | 1.00→1.00 | 22.2→22.5 | 4.5→6.3 | .63→.58 |
| misspunish | 1.00→.87 | 21.6→20.0 | 13.2→13.5 | .62→.55 |
| strafepress| .80→.90 | 19.2→20.3 | 15.2→14.2 | .52→.60 |
| mixed      | .87→.90 | 20.0→20.8 | 14.7→14.6 | .58→.58 |
| gspace     | —→1.00 | —→22.9 | —→14.3 | —→.61 |
| gantiair   | —→.90 | —→20.4 | —→15.3 | —→.58 |
| greengage  | —→.90 | —→20.2 | —→15.3 | —→.56 |
| gmixed     | —→1.00 | —→22.9 | —→14.3 | —→.61 |

- Situational jump (`logs/phase275_behavior.json`): 1.0 far / 1.0 near /
  1.0 hurt / 1.0 post-miss on 12/14 profiles — FLAT. Strafe 0.16–0.39 → 0.0.
- 49/49 tests pass. Bench idle ≈ phase2 (10k/12.5k/20.8k t/s); n_envs=2 kept.

## Phase 4 (utility framework; stages 0–1 trained, reward UNCHANGED)
- Architecture: `sandbox/items/` (registry, inventory/hotbar, effects,
  use-pipeline) + projectiles + battlefield (blocks/webs/fluids) + shield in
  melee + obs v4 (73-dim; v1–v3 intact) + 9-branch action
  (move/sprint/jump/attack/block/use/slot/yaw/pitch) + curriculum unlocks.
- Champ = `checkpoints/phase4_s01/best_model.zip` (1.2M: melee-only 300k,
  shield 900k). Master 9-stage config at `configs/phase4_utility.yaml`.
- Regression (`logs/regression_s01.json`): melee 80% / 21.0 dmg (retained);
  shield-defense 90%, taken 5.0 (vs 14.25 locked), blocks 2.0/ep;
  sustain 100%, gapple used 0.8/ep.
- Shield timing: P(block)=0.37 overall, lower when foe on cooldown (drop to
  counter) — correct direction. Jump still 0.92, strafe 0, crit 1.0.
- Gap: anti-shield offense missing (0% vs shieldholder, 0.0 dealt — attacks
  into raised shield, no reposition/punish).
- 72/72 tests pass. Bench: utility costs ~10% (11k@2env); n_envs=2 kept.

## Phase 4.1 (anti-shield, 1.3M steps, reward UNCHANGED)
- Champ = `checkpoints/phase4_1b/best_model.zip` (7-stage shield curriculum).
  Stage-1 checkpoints untouched. 5 counter opponents (turtle/baiter/strafer/
  pressure/mixed, all imperfect) + shield regression tests.
- Final head-to-head, same seeds (s01 → 1b):
  melee 1.00→1.00 · aggressive .95→.95 · shieldholder .00→.00 (taken 23.9→1.0)
  · turtle .00→.00 (26.5→0.0) · baiter .95→1.00 · strafer .00→.00 (28.2→0.8)
  · pressure .00→.00 (20.5→0.0) · mixed .35→.85 (11.2→23.0 dealt).
- Conditionality (1b mixed replay, WIN): P(attack|shield) 0.07 vs open 0.42;
  P(reposition|blocked) 1.0; drop-punish 16.2/2 drops; shield_frac 0.43.
- Caveats: pure turtles are survival-draws, not wins; strafed attacks go
  left only (12–0); flank angle still frontal (0.15).
- Regression: melee/shield-defense/sustain all 100%. 80/80 tests. Bench
  unchanged (11k@2env); n_envs=2 kept.
- Gate: **OPTION A with H-partial** — counter-shield fundamentals learned
  (stop waste, recognize, reposition, punish openings, retain all) with
  unchanged reward; lateral symmetry + turtle offense are the named follow-ups.

## Phase 4.1H (turtle offense + symmetry, 1.2M steps, reward UNCHANGED)
- Champ = `checkpoints/phase4_1h/best_model.zip`. Turtles A/B/C (long holds,
  observable windows, delayed re-raise) + circlerL/R + speederL/R probes.
- Turtle offense LEARNED: A/B/C all 1.00 wins, 22.2 dealt, 0–4.8 taken.
  holder 0.23 wins (was 0.00), taken 3.0 (was 23.9).
- Symmetry: bias flipped L→R (still 100/0, not fixed). Mirrored circlers
  0.93/0.90 (symmetric performance). Diagnosis: orbit speed (100°+/tick) dwarfs
  any believable shield rotation, and yaw tracking nullifies lateral offset —
  so L/R carries almost no payoff differential (0.57 vs 0.67 hit at scripted
  level). Pressure exists but is weak.
- Regression 100% (shield-defense taken 2.25, blocks 2.3). 85/85 tests.
  Bench unchanged; n_envs=2 kept.
- Gate: **OPTION B** — turtle offense done; lateral selection needs stronger
  directional pressure (speeder foes registered, circler-heavy stage proposed).
  No reward/architecture change indicated: the information exists in obs.

## Phase 4 Stage 2 (sustain; reward UNCHANGED) — OPTION B
- Stages 2A (bread) → 2B (gapple) → 2C (pressure) → 2D (lean) → 2E/2E2/2E3.
  Champ = `checkpoints/phase4_stage2/2e3/best_model.zip`.
- 2E3 suite (30 eps): melee 1.00, aggressive .87, shieldmixed 1.00,
  circlers 1.00/1.00, punish 1.00, mixedeater 1.00, turtleB 1.00.
  Hit .64–.76, taken mostly <8. Rebalanced rehearsal fixed consolidation.
- Sustain verdict: 2A proved conditional eating (0/0.2/0.6 across
  full-HP/threat/safe); consolidated champs eat ~0 everywhere INCLUDING
  gapple-only low-HP. Counterfactual: scripted eat-first at 4 HP wins 1/10
  vs fight-only 5/10 — skipping is currently RATIONAL (short TTK, 32-tick
  vulnerability). Sustain has no marginal value while win rate ≈1.00.
- Gate: OPTION B — skill is representable (2A) but unneeded at this level.
  Next: create ADVERSARIAL NEED (attrition/longer fights where no-heal loses),
  not more same-distribution training. 86/86 tests. Reward byte-identical.

## Phase 4 Stage 3A (pearl mobility; reward UNCHANGED) — OPTION B
- Decisive scenario found by elimination: flat chase/escape/wall-top all fail
  (equal speeds corner; 75° arc reaches wall-tops). Kite-cutoff works: 32-arena
  + divider wall + wall-smart kiter → scripted pearl 9/10 wins vs 10/10 draws.
- Champ = `checkpoints/phase4_stage3/3a2/final.zip` (3A 300k + 3A2 300k).
- Strict gate (locks, kite arena, matched mechanics): kite 0.45 vs 0.20
  no-pearl (9/20 vs 4/20, p≈0.18 — positive, noisy); melee 1.00; mixed 0.95;
  defense/sustain 1.0; symmetry flip intact.
- Behavior: throws 0.1–0.7/ep vs kiter only (conditional, conserved);
  3A→3A2 shows the familiar interference seesaw (throw rate vs melee).
- Eval-harness fixes: suite now supports --enabled/--loadout/--arena-size/
  --obstacles/--mechanics; earlier full-unlock suite numbers allowed locked-out
  items (spontaneous cobweb use observed — noted, out of scope).
- Gate: OPTION B — chase-cutoff emerged but noisy; disengage/re-engage/
  terrain/resource-economics unproven. Next: 3B escape-to-eat (low HP +
  aggressive, open arena — no walls needed: 15-block gap > 32-tick eat).
  94/94 tests. Reward byte-identical. Pearl stays unlocked; nothing else opens.
## Phase 4 Stage 2.6 (adversarial sustain need) — OPTION A (narrow)
- Decisive experiment: regen ruled INERT (identical 63-tick fights ±regen;
  delay exceeds fight length), so per minimal-intervention it was skipped for
  training. The active ingredient is low-HP starts alone (scripted eat-first
  0→9/10 vs fight-first).
- Champ = `checkpoints/phase4_stage26/26a/best_model.zip` (300k, starts 4–10).
- Gate: probe full-HP 0 / threat 0.0 / safe 0.4 eats; gapple-only safe 1/5,
  threat 0/5; symmetry flip + mirror intact; regression melee .95–1.0,
  shield-defense 1.0, shieldmixed 1.00 (recovered), eaters .80–.95,
  regen profiles inert (.95–1.00, no distortion).
- Sustain use is sparse but correctly placed (safe-only); threat discipline
  holds. 90/90 tests. Reward byte-identical. Stage 2 CLOSED → Stage 3 authorized.

## Phase 4.1H-3 (sequential consolidation; reward UNCHANGED) — OPTION A (narrow)
- Question: can shield + circler skills consolidate without wide-pool soup?
- Matrix (win rate, 20-30 eps; base=1h2 champ):

| profile | base | A:S→C | B:C→S | micro: B+10% rehearsal |
| holder | .05 | .00 | .30 | .55 |
| turtleA/B/C | 1/1/.85 | 1/1/.95 | .9/1/.85 | 1/1/.95 |
| baiter | .85 | .75 | .55 | .65 |
| pressure | .00 | .00 | .05 | .30 |
| mixed | .20 | .15 | .40 | .50 |
| circlerL/R | .90/.85 | .75/.85 | .55/.95 | .90/1.00 |
| melee | 1.00 | .90 | .60 | .80 |

- Wide-pool counter-run REJECTED (mixed .13, global hit collapse). Sequential
  legs churn (last-leg dominance both ways). 10% micro-rehearsal holds the
  line: mixed .40→.50, holder .30→.55, circlers ≥.90, symmetry flip intact
  (L-vs-R-foe mirrored), turtles 1.0. Known cost: melee .60–.80 softness.
- Mechanism: per-episode weighted foe sampling (`set_foe_pool`, deterministic).
- 85/85 tests pass. Checkpoints: `phase4_1h3/{Ashield,Acircler,Bcircler,Bshield,micro}/`.
  Reward/obs/arch unchanged. Stage 2 unlock authorized by this gate.

## Phase 4.1H-2 (directional selection; reward UNCHANGED)
- Champ = `checkpoints/phase4_1h2/best_model.zip` (1.2M circler curriculum).
  Turtles 1.00/1.00/0.90; circlerL/R 0.93/0.93; switcher 1.00; melee 0.97.
- Symmetry probe (10 eps × 4 conditions): vs circlerL → R 0.10 (L 0.00);
  vs circlerR → L 0.15 (R 0.00); mirrored pair flips identically. Side
  selection is geometry-dependent (matches foe rotation sense), not 50/50.
- Key mechanism finding: yaw correction (±30°/t) + 75° cone erase pure
  lateral offsets, so payoff lives in co- vs counter-rotation stability
  (probe: taken 0 vs 10) — not in flank side per se.
- Retention counter-run (500k, 7-foe pool) REJECTED by evidence: shieldmixed
  0.20→0.13, hit rates collapsed everywhere. Wide-pool mixing dilutes;
  keeper = phase4_1h2 champ. 85/85 tests pass.
- Gate: **OPTION B** — selection demonstrated; shieldmixed retention needs
  sequential consolidation (not wider pools). No reward/arch change indicated.

## Phase 3 (mechanics ablation; reward UNCHANGED; old checkpoints intact)
- Ablation framework: `CombatConfig` gains `attack_arc_vertical_deg` (=75 →
  legacy single-cone path bit-identical) + `landing_attack_delay_ticks` (=0 →
  no-op); `air_accel` was already a param. Every run labeled in run_config.json.
- Screens 400k, seed 51, same pool (gspace/gantiair/strafer/aggressive ×10):
  base 1.0/1.0/1.0/.90 · A(crit1.0) .50/.20/.70/.50, hit as low as .04 ·
  B(air1.0) acc halved (.28–.31), no behavior change · C(arcV45) acc down
  (.11–.36), still jumps 100% · D(land4) 1.00/1.00/1.00/.90, acc UP (.61–.71),
  strafe re-emerged (0.13–0.34, strafe-when-foe-cd up to 0.5).
- Full 1.2M base vs D, 14 profiles × 30 (`logs/mech_base_full.json`,
  `logs/mech_D_full.json`): D wins/ties 11/14, damage capped at 22.5 almost
  everywhere, taken lower (defensive 3.1, landpunish 1.7), hit .61–.75 vs
  .51–.60. Costs: strafer/aggressive/random −6–7pp (noise-adjacent).
- Conditionality (ep0 replays): D jumps far 1.0 but near 0.71–0.92 (grounded
  in-range attacks 8–29%), strafe 0–0.29 by foe; base is flat 0.5–0.7 jump,
  ~0 strafe. Grounded exchanges still small (3–5%) — minority, not norm.
- Verdict: **OPTION A, qualified** — D (4-tick landing recovery) is the new
  mechanics baseline (`configs/mech/D_land4.yaml`). Crits stay meaningful
  (A proves removing them collapses learning); air/arc knobs change accuracy,
  not strategy. 55/55 tests; bench unchanged; n_envs=2 kept.

## Success criteria
1. Two players fight correctly — yes (`tests/test_combat.py`).
2. Deterministic — yes (`tests/test_determinism.py`).
3. Gymnasium env works — yes (`PvPEnv`, Box[39] / MultiDiscrete).
4. Random-agent episodes — yes (`tools/smoke_random.py`).
5. Scripted opponents — yes (dummy / melee / strafe).
6. PPO runs — yes, 300+ fps headless, 4 parallel envs.
7. Checkpoint save/load — yes (+ resume flag).
8. Metrics visible — tensorboard + eval action histograms.
9. Replays inspectable — `replay/viewer.py` (play/pause/speed/reset/state).
10. Headless parallel — yes (SubprocVecEnv, no pygame import in training path).
11. Ready for melee-intelligence phase — yes (`FUTURE.py`, `PolicyOpponent`).

## Layout
- `sandbox/` — world, physics, entities, combat, simulation (no RL imports)
- `rl/` — obs schema, action map, rewards, gym env, opponents, training
- `replay/` — recorder (npz) + pygame viewer
- `visualization/` — live pygame debug view (optional, headless training unaffected)
- `configs/` — default + PPO hyperparams
- `tests/` — physics/combat/determinism/smoke
- `tools/` — smoke scripts, eval, record demo

## Interfaces (future-proof)
- Policy sees `obs v1` (Box[39]) and emits `MultiDiscrete[5,2,2,2,3,3]`; `action_map.py` translates to controller dict.
- Same controller dict drives sandbox players AND (later) a NeoForge 1.21.1 adapter.
- `Opponent` interface accepts policy checkpoints, history, or scripted bots → self-play ready.
- Stubs reserved (not implemented): pearl/cobweb/lava/potions/inventory/projectiles/LSTM/curriculum.
