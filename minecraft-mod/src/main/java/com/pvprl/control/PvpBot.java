package com.pvprl.control;

import com.mojang.authlib.GameProfile;
import com.pvprl.bridge.BridgeClient;
import com.pvprl.obs.BotFrame;
import com.pvprl.obs.ObservationEncoder;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.effect.MobEffects;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.MoverType;
import net.minecraft.world.entity.projectile.ThrownEnderpearl;
import net.minecraft.world.item.Items;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;
import net.neoforged.neoforge.common.util.FakePlayerFactory;

import java.util.UUID;

/** AI combatant: a server-side FakePlayer, so movement, damage, knockback,
 *  shield blocking, item effects and cooldowns are REAL Minecraft mechanics.
 *  The AI only chooses actions (via the Python bridge); Minecraft resolves
 *  everything. Not a network player (that is a future phase).
 *
 *  Sim-frame conventions (engine.py / action_map.py): yaw 0 = +X, pitch + =
 *  up, wish (moveX right+, moveZ forward+). MC mapping: simYaw = mcYaw + 90,
 *  simPitch = -mcXRot; deltas transfer 1:1 (d(sim)/d(mc) = +1). The policy is
 *  queried every 4 server ticks (= training frame_skip) and yaw/pitch deltas
 *  are applied ONCE per decision to preserve the trained aim dynamics. */
public class PvpBot {
    public static final int DECISION_EVERY = 4;

    private final ServerLevel level;
    private final ServerPlayer bot;
    private final BridgeClient bridge;
    private final ObservationEncoder encoder;
    public final BotFrame me = new BotFrame();
    public final BotFrame foe = new BotFrame();
    public ActionDecoder.Controller lastCtrl;
    public int[] lastAction;
    public double lastInferenceMs = -1, lastBridgeMs = -1, lastRoundTripMs = -1;
    public boolean running;
    public long tick;
    private double prevX, prevZ, prevY;
    private boolean drawingBow;
    private long lastBowShotTick = -1000;

    public PvpBot(ServerLevel level, BridgeClient bridge,
                  ObservationEncoder encoder, BlockPos spawn) {
        this.level = level;
        this.bridge = bridge;
        this.encoder = encoder;
        this.bot = FakePlayerFactory.get(level, new GameProfile(
                UUID.randomUUID(), "pvpbot"));
        this.bot.teleportTo(spawn.getX() + 0.5, spawn.getY(), spawn.getZ() + 0.5);
        level.addFreshEntity(bot);
        snapshot(me, bot, spawn.getX() + 0.5, spawn.getY(), spawn.getZ() + 0.5);
        prevX = me.x; prevY = me.y; prevZ = me.z;
    }

    public ServerPlayer entity() {
        return bot;
    }

    public void remove() {
        running = false;
        bot.discard();
    }

    /** One server tick. foePos/foeVel supplied by the arena (scripted foe). */
    public void tick(long tick, ScriptedOpponent opp) {
        this.tick = tick;
        if (bot.isRemoved()) return;
        // velocities from per-tick deltas (blocks/sec)
        me.vx = (bot.getX() - prevX) * 20.0;
        me.vy = (bot.getY() - prevY) * 20.0;
        me.vz = (bot.getZ() - prevZ) * 20.0;
        prevX = bot.getX(); prevY = bot.getY(); prevZ = bot.getZ();
        snapshot(me, bot, bot.getX(), bot.getY(), bot.getZ());
        me.pushPos(me.x, me.y, me.z);
        opp.snapshotInto(foe);
        foe.pushPos(foe.x, foe.y, foe.z);
        if (me.comboTimer > 0 && --me.comboTimer == 0) me.combo = 0;
        if (foe.comboTimer > 0 && --foe.comboTimer == 0) foe.combo = 0;

        if (running && tick % DECISION_EVERY == 0) decide(tick, opp);
        if (lastCtrl != null && running) apply(lastCtrl, opp);
        else holdNeutral();
    }

    private void decide(long tick, ScriptedOpponent opp) {
        double[] proj = nearestProjectile();
        double[] obs = encoder.encode(me, foe, tick, proj, proj == null ? 0 : 1);
        long t0 = System.nanoTime();
        String line = bridge.query(tick, obs);
        lastRoundTripMs = (System.nanoTime() - t0) / 1e6;
        if (line == null) return; // fallback: keep last controls, camera held
        int[] a = parseAction(line);
        ActionDecoder.Controller c = a == null ? null : ActionDecoder.decode(a);
        if (c == null) return; // malformed/unsafe: ignore, hold course
        lastAction = a;
        lastCtrl = c;
        // aim applied ONCE per decision (trained dynamics, frame_skip = 4)
        bot.setYRot((float) ObservationEncoder.wrapYaw(bot.getYRot() + c.yawDelta));
        bot.setXRot((float) Math.max(-90.0,
                Math.min(90.0, bot.getXRot() - c.pitchDelta)));
        // sim pitch + = up, MC XRot + = down, hence the minus sign.
        if (c.attack) {
            me.lastAttemptTick = tick;
            boolean hit = opp.inMeleeReach(bot);
            me.lastAttemptHit = hit ? 1 : 0;
        }
    }

    private void apply(ActionDecoder.Controller c, ScriptedOpponent opp) {
        // --- rotation already applied in decide(); movement in MC space ---
        double sy = Math.toRadians(me.simYawDeg);
        double fx = Math.cos(sy), fz = Math.sin(sy); // sim fwd in world XZ
        double rx = -fz, rz = fx;                     // sim right in world XZ
        double wx = fx * c.moveZ + rx * c.moveX;
        double wz = fz * c.moveZ + rz * c.moveX;
        double speed = (c.sprint && c.moveZ > 0) ? 5.6 : 4.3; // phys_cfg
        Vec3 d = bot.getDeltaMovement();
        double vy = d.y;
        if (c.jump && bot.onGround()) vy = 0.44; // ~sim jump_velocity 8.8 u/s
        bot.setDeltaMovement(wx * speed / 20.0, vy, wz * speed / 20.0);
        bot.move(MoverType.SELF, bot.getDeltaMovement());

        // --- hotbar select (2-tick switch cost is enforced by the sim-trained
        // --- timing, not re-simulated; selection itself is instant in MC) ---
        if (bot.getInventory().selected != c.slot) {
            if (drawingBow) { bot.releaseUsingItem(); drawingBow = false; }
            bot.getInventory().selected = c.slot;
        }
        // --- attack takes precedence over block (sim.py) ---
        if (c.attack) {
            if (bot.isBlocking()) bot.stopUsingItem();
            if (opp.inMeleeReach(bot)) bot.attack(opp.entity());
        } else if (c.block && hasShield()) {
            ensureOffhandShield();
            if (!bot.isBlocking()) bot.startUsingItem(InteractionHand.OFF_HAND);
        } else if (bot.isBlocking()) {
            bot.stopUsingItem();
        }
        // --- item use through the real MC pipeline ---
        if (c.use && !c.attack) onUse(tick);
        else if (!c.use && drawingBow) { bot.releaseUsingItem(); drawingBow = false; }
        // sim interrupts eating on damage (use.tick_eating); MC does not, so:
        if (bot.isUsingItem() && !drawingBow && bot.hurtTime > 0) bot.stopUsingItem();
    }

    private void onUse(long tick) {
        int slot = bot.getInventory().selected;
        var stack = bot.getInventory().getItem(slot);
        if (stack.isEmpty()) return;
        var item = stack.getItem();
        if (item == Items.ENDER_PEARL) {
            throwPearl();
            me.pearlCooldown = 20;
            stack.shrink(1);
        } else if (item == Items.BOW && bot.getInventory().hasAnyOf(java.util.Set.of(Items.ARROW))) {
            if (!drawingBow && tick - lastBowShotTick >= 15) {
                bot.startUsingItem(InteractionHand.MAIN_HAND);
                drawingBow = true;
            }
        } else if (item.getFoodProperties(stack, bot) != null) {
            if (!bot.isUsingItem()) bot.startUsingItem(InteractionHand.MAIN_HAND);
        }
        // cobweb/block/strength_potion: selected but agent-placed via MC
        // mechanics in a later phase; policy interface keeps the slot.
    }

    private void throwPearl() {
        ThrownEnderpearl p = new ThrownEnderpearl(EntityType.ENDER_PEARL, level);
        p.setOwner(bot);
        p.setPos(bot.getX(), bot.getEyeY(), bot.getZ());
        p.shootFromRotation(bot, bot.getXRot(), bot.getYRot(), 0.0F, 1.5F, 0.0F);
        level.addFreshEntity(p);
    }

    private void holdNeutral() {
        bot.setDeltaMovement(0, bot.getDeltaMovement().y, 0);
        if (bot.isBlocking()) bot.stopUsingItem();
    }

    private boolean hasShield() {
        return bot.getInventory().hasAnyOf(java.util.Set.of(Items.SHIELD))
                || bot.getOffhandItem().is(Items.SHIELD);
    }

    private void ensureOffhandShield() {
        if (bot.getOffhandItem().is(Items.SHIELD)) return;
        var inv = bot.getInventory();
        for (int i = 0; i < inv.getContainerSize(); i++) {
            if (inv.getItem(i).is(Items.SHIELD)) {
                bot.setItemInHand(InteractionHand.OFF_HAND, inv.getItem(i).copy());
                break;
            }
        }
    }

    private double[] nearestProjectile() {
        var box = new AABB(bot.blockPosition()).inflate(24);
        var best = level.getEntitiesOfClass(
                net.minecraft.world.entity.projectile.Projectile.class,
                box, e -> !e.isRemoved()).stream()
                .min((a, b) -> Double.compare(a.distanceToSqr(bot), b.distanceToSqr(bot)))
                .orElse(null);
        if (best == null) return null;
        return new double[]{best.getX() - bot.getX(),
                best.getY() - bot.getY(), best.getZ() - bot.getZ()};
    }

    public static void snapshot(BotFrame f, ServerPlayer p,
                                double x, double y, double z) {
        f.x = x; f.y = y; f.z = z;
        f.mcYaw = p.getYRot(); f.mcPitch = p.getXRot();
        f.simYawDeg = ObservationEncoder.simYaw(p.getYRot());
        f.simPitchDeg = -p.getXRot();
        f.health = p.getHealth(); f.maxHealth = p.getMaxHealth();
        f.onGround = p.onGround();
        f.attackCooldownTicks =
                (int) Math.round((1.0 - p.getAttackStrengthScale(0.5f)) * 20.0);
        f.hurtTime = p.hurtTime;
        f.shieldUp = p.isBlocking();
        f.eating = p.isUsingItem();
        f.absorption = p.getAbsorptionAmount();
        f.hunger = p.getFoodData().getFoodLevel();
        f.speedTicks = effectTicks(p, MobEffects.MOVEMENT_SPEED);
        f.strengthTicks = effectTicks(p, MobEffects.DAMAGE_BOOST);
        f.selectedSlot = p.getInventory().selected;
        var inv = p.getInventory();
        f.counts[0] = count(inv, Items.IRON_SWORD, Items.DIAMOND_SWORD, Items.STONE_SWORD);
        f.counts[1] = count(inv, Items.SHIELD);
        f.counts[2] = count(inv, Items.BREAD);
        f.counts[3] = count(inv, Items.GOLDEN_APPLE);
        f.counts[4] = count(inv, Items.ENDER_PEARL);
        f.counts[5] = count(inv, Items.COBWEB);
        f.counts[6] = count(inv, Items.DIRT, Items.COBBLESTONE);
        f.counts[7] = count(inv, Items.SPLASH_POTION);
        f.counts[8] = count(inv, Items.BOW) > 0 ? count(inv, Items.ARROW) : 0;
        f.pearlCooldown = p.getCooldowns().isOnCooldown(Items.ENDER_PEARL) ? 10 : 0;
        f.gappleCooldown = 0; // MC has none; mod tracks via BotState if needed
        f.eatingTicksLeft = p.isUsingItem() ? 16.0 : 0.0; // ~half of 32-tick sim use
    }

    private static int effectTicks(ServerPlayer p, net.minecraft.core.Holder<net.minecraft.world.effect.MobEffect> e) {
        var inst = p.getEffect(e);
        return inst == null ? 0 : inst.getDuration();
    }

    private static int count(net.minecraft.world.entity.player.Inventory inv,
                             net.minecraft.world.item.Item... items) {
        int n = 0;
        for (int i = 0; i < inv.getContainerSize(); i++) {
            var s = inv.getItem(i);
            for (var it : items) if (s.is(it)) n += s.getCount();
        }
        return n;
    }

    /** Minimal int-array parse of the bridge response (Gson-free fallback). */
    static int[] parseAction(String line) {
        try {
            int i = line.indexOf("\"action\"");
            if (i < 0) return null;
            int a = line.indexOf('[', i), b = line.indexOf(']', a);
            if (a < 0 || b < 0) return null;
            String[] parts = line.substring(a + 1, b).split(",");
            int[] out = new int[parts.length];
            for (int k = 0; k < parts.length; k++) out[k] = Integer.parseInt(parts[k].trim());
            com.pvprl.debug.DebugState.noteInference(line);
            return out;
        } catch (Exception e) {
            return null;
        }
    }
}
