"""Pygame replay viewer: play/pause/speed/reset + state inspection."""
from __future__ import annotations

import argparse
import math

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default="replays/episode_latest.npz")
    args = ap.parse_args()
    import pygame
    from replay.recorder import load_replay
    frames, meta = load_replay(args.replay)
    n = len(frames["tick"])
    print(f"loaded {n} ticks, meta={meta}")
    size = float(meta.get("size", 20))
    W, H, PAD = 640, 640, 40
    pygame.init()
    screen = pygame.display.set_mode((W + 260, H))
    pygame.display.set_caption("PvP replay")
    font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()
    i, playing, speed = 0, True, 1
    running = True

    def world_xy(x, z):
        s = (W - 2 * PAD) / size
        return PAD + x * s, PAD + z * s

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    playing = not playing
                elif e.key == pygame.K_r:
                    i = 0
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(8, speed + 1)
                elif e.key == pygame.K_MINUS:
                    speed = max(1, speed - 1)
                elif e.key == pygame.K_RIGHT:
                    i = min(n - 1, i + 1)
                elif e.key == pygame.K_LEFT:
                    i = max(0, i - 1)
        if playing:
            i = min(n - 1, i + speed)
        screen.fill((12, 12, 16))
        # arena
        x0, y0 = world_xy(0, 0)
        x1, y1 = world_xy(size, size)
        pygame.draw.rect(screen, (40, 40, 48), (x0, y0, x1 - x0, y1 - y0), 2)
        for k, color in (("p0", (90, 160, 255)), ("p1", (255, 110, 110))):
            px, pz = float(frames[f"{k}_pos"][i][0]), float(frames[f"{k}_pos"][i][2])
            vx, vz = float(frames[f"{k}_vel"][i][0]), float(frames[f"{k}_vel"][i][2])
            yaw = float(frames[f"{k}_yaw"][i])
            cx, cy = world_xy(px, pz)
            pygame.draw.circle(screen, color, (int(cx), int(cy)), 9)
            yr = math.radians(yaw)
            ex, ey = cx + math.cos(yr) * 22, cy + math.sin(yr) * 22
            pygame.draw.line(screen, color, (cx, cy), (ex, ey), 2)
            pygame.draw.line(screen, (200, 200, 200), (cx, cy), (cx + vx * 4, cy + vz * 4), 1)
        lines = [
            f"tick {int(frames['tick'][i])}/{n} {'PLAY' if playing else 'PAUSE'} x{speed}",
            f"p0 hp={float(frames['p0_hp'][i]):.1f} atk={bool(frames['a0_attack'][i])}",
            f"p1 hp={float(frames['p1_hp'][i]):.1f} atk={bool(frames['a1_attack'][i])}",
            f"rew={float(frames['reward'][i]):+.3f}",
            f"winner={meta.get('winner')} dmg={meta.get('damage')}",
            f"p0 shield={bool(frames['sh0'][i]) if 'sh0' in frames else '?'} "
            f"abs={float(frames['abs0'][i]):.0f} sel={int(frames['sel'][i])} "
            f"eat={int(frames['eat0'][i])} fx={int(frames['fx0'][i])} "
            f"proj={int(frames['nproj'][i])} blk={int(frames['nblocks'][i])} "
            f"web={int(frames['nwebs'][i])}" if 'sh0' in frames else "pre-utility replay",
            "[space] play/pause  [+-] speed",
            "[<-/->] step  [R] reset",
        ]
        for j, t in enumerate(lines):
            screen.blit(font.render(t, True, (220, 220, 220)), (W + 12, 20 + j * 22))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
