"""Live pygame debug view: positions, hitboxes, range, vectors, rewards, action/obs."""
from __future__ import annotations

import argparse
import math

import numpy as np


_state = {}


def _screen():
    import pygame
    if not _state.get("screen"):
        pygame.init()
        W, H = 640, 640
        _state["screen"] = pygame.display.set_mode((W + 300, H))
        _state["font"] = pygame.font.SysFont("consolas", 14)
        _state["clock"] = pygame.time.Clock()
    return _state["screen"], _state["font"], _state["clock"]


def render_sim(sim, reward_comp=None, action=None, obs=None) -> None:
    import pygame
    screen, font, clock = _screen()
    W, H, PAD = 640, 640, 40
    size = float(sim.world.size)
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            raise SystemExit
    screen.fill((12, 12, 16))

    def wxy(x, z):
        s = (W - 2 * PAD) / size
        return PAD + x * s, PAD + z * s

    x0, y0 = wxy(0, 0)
    x1, y1 = wxy(size, size)
    pygame.draw.rect(screen, (40, 40, 48), (x0, y0, x1 - x0, y1 - y0), 2)
    for o in sim.world.obstacles:
        ax, ay = wxy(o.x0, o.z0)
        bx, by = wxy(o.x1, o.z1)
        pygame.draw.rect(screen, (70, 70, 80), (ax, ay, bx - ax, by - ay))
    # attack range of p0
    p0 = sim.players[0]
    cx, cy = wxy(p0.pos[0], p0.pos[2])
    s = (W - 2 * PAD) / size
    pygame.draw.circle(screen, (60, 90, 140), (int(cx), int(cy)), int(3.0 * s), 1)
    for i, p in enumerate(sim.players):
        color = (90, 160, 255) if i == 0 else (255, 110, 110)
        cx, cy = wxy(p.pos[0], p.pos[2])
        pygame.draw.circle(screen, color, (int(cx), int(cy)), 9)
        yr = math.radians(p.yaw)
        pygame.draw.line(screen, color, (cx, cy), (cx + math.cos(yr) * 24, cy + math.sin(yr) * 24), 2)
        pygame.draw.line(screen, (200, 200, 200), (cx, cy),
                         (cx + p.vel[0] * 4, cy + p.vel[2] * 4), 1)
    # target direction p0->p1
    d = sim.players[1].pos - sim.players[0].pos
    ax, ay = wxy(*sim.players[0].pos[[0, 2]])
    bx, by = wxy(*sim.players[1].pos[[0, 2]])
    pygame.draw.line(screen, (120, 220, 120), (ax, ay), (bx, by), 1)

    lines = [f"tick {sim.tick} winner={sim.winner}",
             f"p0 hp={sim.players[0].health:.1f} cd={sim.players[0].attack_cooldown}",
             f"p1 hp={sim.players[1].health:.1f} cd={sim.players[1].attack_cooldown}",
             f"dist={sim.distance():.2f} dmg={sim.damage_dealt}"]
    if reward_comp:
        lines.append("rew " + " ".join(f"{k}={v:+.2f}" for k, v in reward_comp.items()))
    if action is not None:
        from rl.environment import action_map as am
        lines.append(f"act {am.describe(action)} {np.asarray(action).tolist()}")
    if obs is not None:
        from rl.environment import obs_schema as ob
        lines.append(ob.obs_summary(obs))
    for j, t in enumerate(lines):
        screen.blit(font.render(t, True, (220, 220, 220)), (W + 10, 20 + j * 20))
    pygame.display.flip()
    clock.tick(30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="melee")
    ap.add_argument("--ticks", type=int, default=600)
    args = ap.parse_args()
    from sandbox.simulation.sim import PvPSim
    from rl.opponents.scripted import create
    from rl.environment import action_map
    rng = np.random.default_rng(0)
    sim = PvPSim(seed=0)
    opp = create(args.opponent)
    for _ in range(args.ticks):
        if sim.done:
            break
        c0 = action_map.to_controller(action_map.random_controller(rng).tolist())
        sim.step(c0, opp.act(sim, 1))
        render_sim(sim)


if __name__ == "__main__":
    main()
