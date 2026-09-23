# PvP RL NeoForge bridge mod (Minecraft 1.21.1, Java 21)

Server-side AI PvP test entity driven by the trained PPO checkpoint over a
localhost TCP bridge. Minecraft owns physics/combat/truth; Python owns inference.

## Build (needs JDK 21 + internet for MDK deps)

```bash
cd minecraft-mod
gradle build        # or ./gradlew build with a wrapper from the NeoForge MDK
```

Copy `build/libs/pvprl-0.1.0.jar` into the server `mods/` folder (NeoForge
21.1.x, Minecraft 1.21.1, singleplayer integrated server works).

## Run order (mirrors §20 TEST A–K)

1. Start Minecraft (no AI yet) — TEST A.
2. `python deployment/inference_server.py` (binds 127.0.0.1:25575) — TEST B/C.
3. In-game: `/pvpbot spawn` → `/pvpbot foe melee` → `/pvpbot start` — TEST F–K.
4. `/pvpbot debug` toggles the action-bar HUD; `/pvpbot status` prints state;
   `/pvpbot dump <tick>` writes one obs/action pair for golden replay
   (`python deployment/replay_golden.py pvp-rl-debug/recordings/`).

Full command list: `spawn remove reset start stop status debug foe <mode> dump <tick> bridge <port>`.
Foe modes: `stationary melee strafe shield`.

## What the bot is

A server-side `FakePlayer` ("pvpbot") + scripted `FakePlayer` foe ("pvpfoe").
Both take REAL damage/knockback/death; shield/food/bow/pearl go through the
real item pipeline. This is a player-like combatant, NOT a network player
(real-player impersonation is an explicitly future phase).
