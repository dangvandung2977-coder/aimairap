package com.pvprl.obs;

import java.util.List;

/** EXACT 73-dim observation v4 encoder. Single source of truth for semantics:
 *  rl/environment/obs_schema.py (build_obs_v1/v2/v3 + v4 tail). Index table:
 *  SELF 0-12, ENEMY 13-25, COMBAT 26-30, WORLD 31-38, V2 39-43, V3 44-46,
 *  V4 47-72. Any deviation from this table is a bug, not a judgment call.
 *
 *  Coordinate frame (must match the sim, else every relative term is wrong):
 *    sim yaw 0 = +X. MC yaw 0 = -Z, MC yaw -90 = +X  =>  simYaw = mcYaw + 90
 *    sim pitch + = up, MC XRot + = down               =>  simPitch = -mcPitch
 *  (engine.look_dir: fwd=(cos yaw, sin pitch, sin yaw); wrap_yaw to (-180,180])
 */
public class ObservationEncoder {
    public static final int OBS_DIM = 73;

    private final double arenaSize;
    private final double lo, hi;
    private final List<double[]> pillars; // x0,z0,x1,z1 obstacle rects

    public ObservationEncoder(double arenaSize, double wallMargin,
                              List<double[]> pillars) {
        this.arenaSize = arenaSize;
        this.lo = wallMargin;
        this.hi = arenaSize - wallMargin;
        this.pillars = pillars;
    }

    public static double wrapYaw(double yaw) {
        while (yaw > 180.0) yaw -= 360.0;
        while (yaw <= -180.0) yaw += 360.0;
        return yaw;
    }

    public static double simYaw(double mcYaw) {
        return wrapYaw(mcYaw + 90.0);
    }

    private static double clip(double x) {
        return Math.max(-5.0, Math.min(5.0, x)); // obs_schema._clip
    }

    /** Builds the 73-vector. me/foe are sim-frame BotFrames, tick is the
     *  server tick, proj is the nearest live projectile rel. pos or null,
     *  nproj the live projectile count. All normalizers copied from schema. */
    public double[] encode(BotFrame me, BotFrame foe, long tick,
                           double[] projRel, int nproj) {
        double s = arenaSize;
        double[] o = new double[OBS_DIM];
        double sy = Math.toRadians(me.simYawDeg);
        double fy = Math.toRadians(foe.simYawDeg);
        // SELF 0-12 (build_obs_v1)
        o[0] = clip(me.x / s); o[1] = clip(me.y / 10.0); o[2] = clip(me.z / s);
        o[3] = clip(me.vx / 10.0); o[4] = clip(me.vy / 10.0); o[5] = clip(me.vz / 10.0);
        o[6] = Math.sin(sy); o[7] = Math.cos(sy);
        o[8] = clip(me.simPitchDeg / 90.0);
        o[9] = clip(me.health / me.maxHealth);
        o[10] = me.onGround ? 1.0 : 0.0;
        o[11] = clip(me.attackCooldownTicks / 12.0);
        o[12] = clip(me.hurtTime / 10.0);
        // ENEMY 13-25
        double dx = foe.x - me.x, dy = foe.y - me.y, dz = foe.z - me.z;
        o[13] = clip(dx / s); o[14] = clip(dy / 10.0); o[15] = clip(dz / s);
        o[16] = clip((foe.vx - me.vx) / 10.0);
        o[17] = clip((foe.vy - me.vy) / 10.0);
        o[18] = clip((foe.vz - me.vz) / 10.0);
        double dyaw = Math.toRadians(foe.simYawDeg - me.simYawDeg);
        o[19] = Math.sin(dyaw); o[20] = Math.cos(dyaw);
        o[21] = clip(foe.simPitchDeg / 90.0);
        o[22] = clip(foe.health / foe.maxHealth);
        o[23] = foe.onGround ? 1.0 : 0.0;
        o[24] = clip(foe.attackCooldownTicks / 12.0);
        o[25] = clip(foe.hurtTime / 10.0);
        // COMBAT 26-30
        double dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        o[26] = clip(dist / s);
        o[27] = clip(me.damageDealt / 20.0);
        o[28] = clip(foe.damageDealt / 20.0);
        o[29] = clip(me.combo / 5.0);
        long tsh = tick - me.lastHitTick; // init 1e9 -> min 100 -> 5.0
        o[30] = clip(Math.min(tsh, 100) / 20.0);
        // WORLD 31-38 (arena.wall_distances + obstacle_sensor)
        o[31] = clip((me.x - lo) / s); o[32] = clip((hi - me.x) / s);
        o[33] = clip((me.z - lo) / s); o[34] = clip((hi - me.z) / s);
        double[] cell = obstacleSensor(me.x, me.z);
        o[35] = cell[0]; o[36] = cell[1]; o[37] = cell[2]; o[38] = cell[3];
        // V2 39-43 (build_obs_v2)
        double bearing = Math.atan2(dz, dx) - Math.toRadians(me.simYawDeg);
        o[39] = Math.sin(bearing); o[40] = Math.cos(bearing);
        o[41] = clip(Math.sqrt(foe.vx * foe.vx + foe.vz * foe.vz) / 10.0);
        double[] old = foe.oldestPos();
        o[42] = clip((foe.x - old[0]) / s);
        o[43] = clip((foe.z - old[2]) / s);
        // V3 44-46 (build_obs_v3)
        o[44] = clip(Math.min(tick - me.lastAttemptTick, 100) / 20.0);
        o[45] = clip(Math.min(tick - foe.lastAttemptTick, 100) / 20.0);
        o[46] = me.lastAttemptHit;
        // V4 47-72 (v4 tail; HOTBAR order = defs.py:76)
        o[47] = clip(me.selectedSlot / 8.0);
        for (int i = 0; i < 9; i++) o[48 + i] = clip(me.counts[i] / 16.0);
        o[57] = clip(me.pearlCooldown / 20.0);
        o[58] = clip(me.gappleCooldown / 100.0);
        o[59] = clip(me.bowCooldown / 15.0);
        o[60] = clip(me.absorption / 8.0);
        o[61] = me.shieldUp ? 1.0 : 0.0;
        o[62] = clip(me.eatingTicksLeft / 32.0);
        o[63] = clip(me.hunger / 20.0);
        o[64] = clip(foe.absorption / 8.0);
        o[65] = foe.shieldUp ? 1.0 : 0.0;
        o[66] = foe.eating ? 1.0 : 0.0;
        o[67] = clip(nproj / 4.0);
        if (projRel != null) {
            o[68] = clip(projRel[0] / 20.0);
            o[69] = clip(projRel[1] / 10.0);
            o[70] = clip(projRel[2] / 20.0);
        }
        o[71] = clip(me.speedTicks / 1200.0);
        o[72] = clip(me.strengthTicks / 1200.0);
        return o;
    }

    /** 4 binary cells: solid pillar block above the floor within 2m in
     *  +X/-X/+Z/-Z (mirrors arena.obstacle_sensor sampling). isSolidAt is
     *  supplied by the mod (world lookup); walls/floor excluded by the
     *  pillar-rect test, exactly like the sim ignores non-pillar solids. */
    public double[] obstacleSensor(double x, double z) {
        double[] out = new double[4];
        double[][] dirs = {{2.0, 0.0}, {-2.0, 0.0}, {0.0, 2.0}, {0.0, -2.0}};
        for (int i = 0; i < 4; i++) {
            for (double t : new double[]{0.75, 1.5, 2.0}) {
                double sx = x + dirs[i][0] * (t / 2.0);
                double sz = z + dirs[i][1] * (t / 2.0);
                if (sx >= lo && sx <= hi && sz >= lo && sz <= hi && isPillar(sx, sz)) {
                    out[i] = 1.0;
                    break;
                }
            }
        }
        return out;
    }

    private boolean isPillar(double x, double z) {
        for (double[] p : pillars) {
            if (p[0] <= x && x <= p[2] && p[1] <= z && z <= p[3]) return true;
        }
        return false;
    }
}
