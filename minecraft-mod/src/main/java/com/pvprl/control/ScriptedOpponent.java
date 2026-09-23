package com.pvprl.control;

import com.mojang.authlib.GameProfile;
import com.pvprl.obs.BotFrame;
import com.pvprl.obs.ObservationEncoder;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.MoverType;
import net.minecraft.world.item.Items;
import net.minecraft.world.phys.Vec3;
import net.neoforged.neoforge.common.util.FakePlayerFactory;

import java.util.UUID;

/** Deterministic scripted sparring partner (integration testing only, not an
 *  AI system). Modes: stationary (never moves/attacks), melee (approach +
 *  swing in range, mirrors rl/opponents/scripted.py MeleeOpponent), strafe
 *  (orbit + close), shield (hold block when the bot is near/attacking). */
public class ScriptedOpponent {
    public enum Mode {STATIONARY, MELEE, STRAFE, SHIELD}

    private final ServerLevel level;
    private final ServerPlayer foe;
    private Mode mode = Mode.STATIONARY;
    private final BotFrame frame = new BotFrame();
    private double prevX, prevY, prevZ;
    private int dir = 1, t;

    public ScriptedOpponent(ServerLevel level, BlockPos spawn) {
        this.level = level;
        this.foe = FakePlayerFactory.get(level, new GameProfile(
                UUID.randomUUID(), "pvpfoe"));
        this.foe.teleportTo(spawn.getX() + 0.5, spawn.getY(), spawn.getZ() + 0.5);
        level.addFreshEntity(foe);
        giveKit();
        PvpBot.snapshot(frame, foe, foe.getX(), foe.getY(), foe.getZ());
        prevX = foe.getX(); prevY = foe.getY(); prevZ = foe.getZ();
    }

    private void giveKit() {
        var inv = foe.getInventory();
        inv.add(new net.minecraft.world.item.ItemStack(Items.IRON_SWORD));
        inv.add(new net.minecraft.world.item.ItemStack(Items.SHIELD));
        foe.setItemInHand(InteractionHand.OFF_HAND,
                new net.minecraft.world.item.ItemStack(Items.SHIELD));
    }

    public ServerPlayer entity() {
        return foe;
    }

    public void setMode(Mode m) {
        this.mode = m;
        if (m != Mode.SHIELD && foe.isBlocking()) foe.stopUsingItem();
    }

    public void remove() {
        foe.discard();
    }

    public boolean inMeleeReach(ServerPlayer bot) {
        return foe.distanceTo(bot) <= 3.0;
    }

    public void snapshotInto(BotFrame f) {
        f.vx = (foe.getX() - prevX) * 20.0;
        f.vy = (foe.getY() - prevY) * 20.0;
        f.vz = (foe.getZ() - prevZ) * 20.0;
        prevX = foe.getX(); prevY = foe.getY(); prevZ = foe.getZ();
        PvpBot.snapshot(f, foe, foe.getX(), foe.getY(), foe.getZ());
    }

    public void tick(ServerPlayer bot) {
        t++;
        if (t % 40 == 0) dir *= -1;
        frame.vx = 0; frame.vz = 0;
        switch (mode) {
            case STATIONARY -> foe.setDeltaMovement(0, foe.getDeltaMovement().y, 0);
            case MELEE -> {
                face(bot);
                double d = foe.distanceTo(bot);
                if (d > 2.8) {
                    forward(1.0, 0.5 * dir);
                } else if (foe.getAttackStrengthScale(0.5f) > 0.9) {
                    foe.attack(bot);
                }
            }
            case STRAFE -> {
                face(bot);
                double d = foe.distanceTo(bot);
                if (d > 3.3) forward(1.0, 0.5 * dir);
                else forward(0.0, 1.0 * dir);
                if (d <= 3.0 && foe.getAttackStrengthScale(0.5f) > 0.9) foe.attack(bot);
            }
            case SHIELD -> {
                face(bot);
                boolean threat = foe.distanceTo(bot) < 4.0;
                if (threat && !foe.isBlocking()) {
                    foe.startUsingItem(InteractionHand.OFF_HAND);
                } else if (!threat && foe.isBlocking()) {
                    foe.stopUsingItem();
                }
            }
        }
        foe.move(MoverType.SELF, foe.getDeltaMovement());
    }

    private void face(ServerPlayer bot) {
        double dx = bot.getX() - foe.getX(), dz = bot.getZ() - foe.getZ();
        foe.setYRot((float) Math.toDegrees(Math.atan2(dz, dx)) - 90.0f);
    }

    private void forward(double z, double x) {
        double y = Math.toRadians(foe.getYRot());
        double fx = -Math.sin(y), fz = Math.cos(y); // MC forward in XZ
        double rx = -fz, rz = fx;                   // MC right in XZ
        foe.setDeltaMovement(new Vec3((fx * z + rx * x) * 4.3 / 20.0,
                foe.getDeltaMovement().y, (fz * z + rz * x) * 4.3 / 20.0));
    }
}
