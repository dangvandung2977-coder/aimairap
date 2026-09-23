package com.pvprl.control;

import com.pvprl.bridge.BridgeClient;
import com.pvprl.obs.ObservationEncoder;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.block.Blocks;

import java.util.List;

/** Flat test arena + bot/foe lifecycle. Default 32x32 (matches the Stage 3A
 *  arena size 32, wall_margin 1.0: lo=1, hi=31) at y=64 near world spawn.
 *  Kit mirrors the training loadout: sword, shield (+offhand copy), bread x5,
 *  gapple, pearls x4, cobweb x4, dirt x16, bow + arrows x16. */
public class ArenaManager {
    public static final int ARENA = 32;
    public static final int FLOOR_Y = 64;

    private final ServerLevel level;
    private final BlockPos origin;
    private final ObservationEncoder encoder;
    private PvpBot bot;
    private ScriptedOpponent foe;

    public ArenaManager(ServerLevel level, BlockPos origin) {
        this.level = level;
        this.origin = origin;
        this.encoder = new ObservationEncoder(ARENA, 1.0, List.of());
    }

    public void build() {
        for (int x = 0; x < ARENA; x++) {
            for (int z = 0; z < ARENA; z++) {
                level.setBlock(origin.offset(x, -1, z),
                        Blocks.QUARTZ_BLOCK.defaultBlockState(), 3);
                if (x == 0 || z == 0 || x == ARENA - 1 || z == ARENA - 1) {
                    for (int h = 0; h < 4; h++) {
                        level.setBlock(origin.offset(x, h, z),
                                Blocks.GLASS.defaultBlockState(), 3);
                    }
                }
            }
        }
    }

    public void spawn(String host, int port) {
        remove();
        build();
        BlockPos bs = origin.offset(8, 0, ARENA / 2);
        BlockPos fs = origin.offset(ARENA - 8, 0, ARENA / 2);
        bot = new PvpBot(level, new BridgeClient(host, port, 500, 500),
                encoder, bs);
        foe = new ScriptedOpponent(level, fs);
        giveKit(bot.entity().getInventory());
        bot.entity().teleportTo(bs.getX() + 0.5, FLOOR_Y, bs.getZ() + 0.5);
        foe.entity().teleportTo(fs.getX() + 0.5, FLOOR_Y, fs.getZ() + 0.5);
    }

    private static void giveKit(net.minecraft.world.entity.player.Inventory inv) {
        inv.clearContent();
        inv.add(new ItemStack(Items.IRON_SWORD));
        inv.add(new ItemStack(Items.SHIELD));
        inv.add(new ItemStack(Items.BREAD, 5));
        inv.add(new ItemStack(Items.GOLDEN_APPLE));
        inv.add(new ItemStack(Items.ENDER_PEARL, 4));
        inv.add(new ItemStack(Items.COBWEB, 4));
        inv.add(new ItemStack(Items.DIRT, 16));
        inv.add(new ItemStack(Items.BOW));
        inv.add(new ItemStack(Items.ARROW, 16));
    }

    public void remove() {
        if (bot != null) { bot.remove(); bot = null; }
        if (foe != null) { foe.remove(); foe = null; }
    }

    public void reset() {
        if (bot == null || foe == null) return;
        bot.entity().setHealth(bot.entity().getMaxHealth());
        foe.entity().setHealth(foe.entity().getMaxHealth());
        bot.entity().teleportTo(origin.getX() + 8.5, FLOOR_Y, origin.getZ() + ARENA / 2);
        foe.entity().teleportTo(origin.getX() + ARENA - 8 + 0.5, FLOOR_Y,
                origin.getZ() + ARENA / 2);
        clearProjectiles();
    }

    private void clearProjectiles() {
        var box = new net.minecraft.world.phys.AABB(
                origin.getX(), origin.getY(), origin.getZ(),
                origin.getX() + ARENA, origin.getY() + 8, origin.getZ() + ARENA);
        for (var p : level.getEntitiesOfClass(
                net.minecraft.world.entity.projectile.Projectile.class, box)) {
            p.discard();
        }
    }

    public PvpBot bot() {
        return bot;
    }

    public ScriptedOpponent foe() {
        return foe;
    }

    public boolean active() {
        return bot != null && foe != null;
    }
}
