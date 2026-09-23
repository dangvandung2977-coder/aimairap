package com.pvprl.obs;

import java.util.ArrayDeque;
import java.util.Deque;

/** Per-combatant bridge state. Mirrors the counters PvPSim keeps per player
 *  (sandbox/simulation/sim.py): damage dealt, combo + timer, last attempt
 *  tick/hit, time-since-hit, and a 4-entry position history (prev_pos). */
public class BotFrame {
    public double x, y, z;
    public double vx, vy, vz; // blocks/sec, from per-tick deltas
    public float mcYaw, mcPitch;
    /** Sim-frame angles, derived each tick (see ObservationEncoder header):
     *  simYawDeg = wrap(mcYaw + 90), simPitchDeg = -mcPitch. */
    public double simYawDeg, simPitchDeg;
    public double health, maxHealth = 20.0;
    public boolean onGround;
    public int attackCooldownTicks; // approximated from attack strength
    public int hurtTime;
    public boolean shieldUp;
    public boolean eating;
    public double absorption;
    public int hunger = 20;
    public int speedTicks, strengthTicks;
    public int selectedSlot;
    public int[] counts = new int[9];
    public int pearlCooldown, gappleCooldown, bowCooldown; // ticks remaining
    public double eatingTicksLeft;
    public double damageDealt;
    public int combo, comboTimer;
    public long lastAttemptTick = -1_000_000_000L;
    public int lastAttemptHit;
    public long lastHitTick = -1_000_000_000L;
    public final Deque<double[]> prevPos = new ArrayDeque<>();

    public BotFrame() {
        for (int i = 0; i < 4; i++) prevPos.addLast(new double[]{0, 0, 0});
    }

    public void pushPos(double x, double y, double z) {
        prevPos.addLast(new double[]{x, y, z});
        while (prevPos.size() > 4) prevPos.removeFirst();
    }

    public double[] oldestPos() {
        return prevPos.peekFirst();
    }

    /** Called when this frame's owner lands a hit (any source of real damage
     *  it dealt, melee or arrow). Mirrors sim.py combo bookkeeping. */
    public void noteHit(long tick, BotFrame victim) {
        damageDealt += 0; // damage totals are accumulated by the caller
        if (comboTimer > 0) combo++; else combo = 1;
        comboTimer = 60;
        victim.combo = 0;
        victim.comboTimer = 0;
        lastHitTick = tick;
    }
}
