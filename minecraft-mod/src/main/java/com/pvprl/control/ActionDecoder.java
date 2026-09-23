package com.pvprl.control;

import java.util.Arrays;

/** EXACT action decoder. Authoritative mapping: rl/environment/action_map.py
 *  to_controller() + sandbox/items/defs.py:76 HOTBAR. NVEC = [5,2,2,2,2,2,9,3,3].
 *
 *  action[0] move   0 stop, 1 forward, 2 back, 3 left, 4 right
 *  action[1] sprint 0/1            action[2] jump   0/1
 *  action[3] attack 0/1 (takes precedence over block, sim.py)
 *  action[4] block  0/1 (shield; 2-tick raise in sim)
 *  action[5] use    0/1 (validated item pipeline, use.py)
 *  action[6] slot   0 sword, 1 shield, 2 bread, 3 golden_apple, 4 ender_pearl,
 *                   5 cobweb, 6 block, 7 strength_potion, 8 bow
 *  action[7] yaw    0 -15 deg, 1 keep, 2 +15 deg (ONE application per
 *                   decision; the policy was trained with frame_skip = 4, so
 *                   the mod queries every 4 server ticks and applies deltas
 *                   once, preserving the trained aim dynamics)
 *  action[8] pitch  0 -10 deg, 1 keep, 2 +10 deg (clamped [-90, 90])
 */
public class ActionDecoder {
    public static final int[] NVEC = {5, 2, 2, 2, 2, 2, 9, 3, 3};
    public static final double YAW_STEP = 15.0;   // action_map.YAW_STEP
    public static final double PITCH_STEP = 10.0; // action_map.PITCH_STEP

    public static class Controller {
        public double moveX, moveZ; // sim frame: +Z fwd, +X right
        public boolean sprint, jump, attack, block, use;
        public int slot;
        public double yawDelta, pitchDelta;
    }

    /** Returns null when illegal (caller holds previous/safe controls). */
    public static Controller decode(int[] a) {
        if (a == null || a.length != NVEC.length) return null;
        for (int i = 0; i < NVEC.length; i++) {
            if (a[i] < 0 || a[i] >= NVEC[i]) return null;
        }
        Controller c = new Controller();
        switch (a[0]) {
            case 1: c.moveZ = 1.0; break;
            case 2: c.moveZ = -1.0; break;
            case 3: c.moveX = -1.0; break;
            case 4: c.moveX = 1.0; break;
            default: break;
        }
        c.sprint = a[1] == 1;
        c.jump = a[2] == 1;
        c.attack = a[3] == 1;
        c.block = a[4] == 1;
        c.use = a[5] == 1;
        c.slot = a[6];
        c.yawDelta = (a[7] - 1) * YAW_STEP;
        c.pitchDelta = (a[8] - 1) * PITCH_STEP;
        return c;
    }

    public static String describe(int[] a) {
        String[] names = {"stop", "fwd", "back", "left", "right"};
        String yaw = new String[]{"L", "-", "R"}[a[7]];
        String extra = (a[4] == 1 ? "B" : "") + (a[5] == 1 ? "U" + a[6] : "");
        return names[a[0]] + (a[1] == 1 ? "S" : "") + (a[2] == 1 ? "J" : "")
                + (a[3] == 1 ? "X" : "") + extra + yaw;
    }

    public static boolean same(int[] a, int[] b) {
        return Arrays.equals(a, b);
    }
}
