"""Future-compat notes (reserved, NOT implemented in phase 1).

How each future feature plugs in without rewriting the simulator:
- pearl/cobweb/lava/potions/food/inventory/projectiles/block-place/terrain:
  new optional fields on controller dict + new systems in sandbox/,
  e.g. {"place_block": bool}. Policy action space grows a branch; OBS_VERSION bumps.
- opponent modeling / LSTM: rl/policies/ — PolicyInterface already hides the model;
  recurrent policies only change train_ppo --policy, not the env.
- curriculum / population self-play: PolicyOpponent + Opponent.create() accept
  checkpoints/history; training script swaps --opponent policy:<path>.
- NeoForge 1.21.1 adapter: implements the same two functions —
  game-state -> obs v1 vector, controller dict -> clicks/keys — so any checkpoint
  saved here loads there unchanged (see rl/policies/interface.py).
"""
