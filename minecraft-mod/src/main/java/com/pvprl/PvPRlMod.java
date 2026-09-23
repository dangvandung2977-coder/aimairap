package com.pvprl;

import com.pvprl.control.ActionDecoder;
import com.pvprl.control.PvpCommands;
import com.pvprl.debug.DebugState;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerPlayer;
import net.neoforged.bus.api.IEventBus;
import net.neoforged.fml.common.Mod;
import net.neoforged.neoforge.common.NeoForge;
import net.neoforged.neoforge.event.RegisterCommandsEvent;
import net.neoforged.neoforge.event.server.ServerStartedEvent;
import net.neoforged.neoforge.event.server.ServerStoppingEvent;
import net.neoforged.neoforge.event.tick.ServerTickEvent;

/** PvP RL bridge mod. Drives the AI bot every server tick; the policy is
 *  queried every 4 ticks (= training frame_skip). Debug HUD goes to the
 *  nearest real player's action bar when /pvpbot debug is on. */
@Mod(PvPRlMod.MOD_ID)
public class PvPRlMod {
    public static final String MOD_ID = "pvprl";
    private MinecraftServer server;

    public PvPRlMod(IEventBus modEventBus) {
        NeoForge.EVENT_BUS.addListener(this::onRegisterCommands);
        NeoForge.EVENT_BUS.addListener(this::onStarted);
        NeoForge.EVENT_BUS.addListener(this::onStopping);
        NeoForge.EVENT_BUS.addListener(this::onTick);
    }

    private void onRegisterCommands(RegisterCommandsEvent e) {
        PvpCommands.register(e.getDispatcher());
    }

    private void onStarted(ServerStartedEvent e) {
        server = e.getServer();
    }

    private void onStopping(ServerStoppingEvent e) {
        if (PvpCommands.arena() != null) PvpCommands.arena().remove();
        server = null;
    }

    private void onTick(ServerTickEvent.Post e) {
        var arena = PvpCommands.arena();
        if (arena == null || !arena.active() || server == null) return;
        long tick = server.getTickCount();
        try {
            arena.foe().tick(arena.bot().entity());
            arena.bot().tick(tick, arena.foe());
            DebugState.noteTick(tick, arena.bot().lastRoundTripMs);
            if (PvpCommands.debugOn() && tick % 10 == 0) hud(arena);
        } catch (Exception ex) {
            // AI bridge unavailable (or any bug): log, keep server alive.
            server.sendSystemMessage(Component.literal(
                    "[pvpbot] AI bridge unavailable: " + ex.getClass().getSimpleName()));
            arena.bot().running = false;
        }
    }

    private void hud(com.pvprl.control.ArenaManager arena) {
        ServerPlayer viewer = null;
        double best = Double.MAX_VALUE;
        for (ServerPlayer p : server.getPlayerList().getPlayers()) {
            if (p instanceof net.neoforged.neoforge.common.util.FakePlayer) continue;
            double d = p.distanceTo(arena.bot().entity());
            if (d < best) { best = d; viewer = p; }
        }
        if (viewer == null) return;
        var b = arena.bot();
        viewer.displayClientMessage(Component.literal(
                String.format("tick=%d hp=%.0f foe=%.0f d=%.1f act=%s infer=%.1fms rtt=%.1fms",
                        b.tick, b.entity().getHealth(), arena.foe().entity().getHealth(),
                        b.entity().distanceTo(arena.foe().entity()),
                        b.lastAction == null ? "none" : ActionDecoder.describe(b.lastAction),
                        b.lastInferenceMs, b.lastRoundTripMs)), true);
    }
}
