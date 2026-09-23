package com.pvprl.control;

import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.pvprl.bridge.BridgeClient;
import com.pvprl.debug.DebugState;
import com.pvprl.obs.ObservationEncoder;
import com.pvprl.debug.DebugState;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.core.BlockPos;
import net.minecraft.network.chat.Component;
import net.minecraft.server.level.ServerLevel;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** /pvpbot spawn|remove|reset|start|stop|status|debug|foe|dump */
public class PvpCommands {
    private static ArenaManager arena;
    private static String host = "127.0.0.1";
    private static int port = 25575;
    private static boolean debug;

    public static void register(CommandDispatcher<CommandSourceStack> d) {
        var root = Commands.literal("pvpbot").requires(s -> s.hasPermission(2));
        root.then(Commands.literal("spawn").executes(c -> {
            ServerLevel level = c.getSource().getLevel();
            BlockPos at = BlockPos.containing(c.getSource().getPosition()).above(1);
            arena = new ArenaManager(level, new BlockPos(at.getX() - 16, 64, at.getZ() - 16));
            arena.spawn(host, port);
            say(c.getSource(), "arena built, bot + foe spawned (AI idle, /pvpbot start)");
            return 1;
        }));
        root.then(Commands.literal("remove").executes(c -> {
            if (arena != null) arena.remove();
            say(c.getSource(), "removed");
            return 1;
        }));
        root.then(Commands.literal("reset").executes(c -> {
            if (arena != null) arena.reset();
            say(c.getSource(), "reset");
            return 1;
        }));
        root.then(Commands.literal("start").executes(c -> {
            if (arena != null && arena.active()) arena.bot().running = true;
            say(c.getSource(), "AI running");
            return 1;
        }));
        root.then(Commands.literal("stop").executes(c -> {
            if (arena != null && arena.active()) arena.bot().running = false;
            say(c.getSource(), "AI stopped (neutral)");
            return 1;
        }));
        root.then(Commands.literal("status").executes(c -> {
            var s = c.getSource();
            if (arena == null || !arena.active()) { say(s, "no arena"); return 1; }
            var b = arena.bot();
            say(s, "running=" + b.running + " hp=" + b.entity().getHealth()
                    + " foeHp=" + arena.foe().entity().getHealth()
                    + " dist=" + String.format("%.2f", b.entity().distanceTo(arena.foe().entity()))
                    + " lastAction=" + (b.lastAction == null ? "none"
                    : ActionDecoder.describe(b.lastAction))
                    + " infer=" + String.format("%.2f", b.lastInferenceMs) + "ms"
                    + " rtt=" + String.format("%.2f", b.lastRoundTripMs) + "ms");
            say(s, DebugState.summary());
            return 1;
        }));
        root.then(Commands.literal("debug").executes(c -> {
            debug = !debug;
            say(c.getSource(), "debug " + (debug ? "ON" : "OFF"));
            return 1;
        }));
        root.then(Commands.literal("foe")
                .then(Commands.argument("mode", StringArgumentType.word()).executes(c -> {
                    String m = StringArgumentType.getString(c, "mode").toUpperCase();
                    if (arena != null && arena.active()) {
                        arena.foe().setMode(ScriptedOpponent.Mode.valueOf(m));
                        say(c.getSource(), "foe=" + m);
                    }
                    return 1;
                })));
        root.then(Commands.literal("dump")
                .then(Commands.argument("tick", IntegerArgumentType.integer()).executes(c -> {
                    say(c.getSource(), dump(c.getSource().getLevel(),
                            IntegerArgumentType.getInteger(c, "tick")));
                    return 1;
                })));
        root.then(Commands.literal("bridge")
                .then(Commands.argument("port", IntegerArgumentType.integer(1, 65535))
                        .executes(c -> {
                            port = IntegerArgumentType.getInteger(c, "port");
                            say(c.getSource(), "bridge port=" + port + " (respawn to apply)");
                            return 1;
                        })));
        d.register(root);
    }

    /** Dumps one live observation/action pair to JSON for golden replay. */
    static String dump(ServerLevel level, int tick) {
        if (arena == null || !arena.active()) return "no arena";
        var b = arena.bot();
        var enc = new ObservationEncoder(32, 1.0, List.of());
        double[] obs = enc.encode(b.me, b.foe, tick, null, 0);
        StringBuilder sb = new StringBuilder("{\"tick\":");
        sb.append(tick).append(",\"observation\":[");
        for (int i = 0; i < obs.length; i++) sb.append(i == 0 ? "" : ",").append(obs[i]);
        sb.append("]}");
        try {
            Path dir = level.getServer().getServerDirectory()
                    .resolve("pvp-rl-debug/recordings");
            Files.createDirectories(dir);
            Path f = dir.resolve(String.format("obs_%06d.json", tick));
            Files.writeString(f, sb.toString());
            return "wrote " + f;
        } catch (Exception e) {
            return "dump failed: " + e.getMessage();
        }
    }

    private static void say(CommandSourceStack s, String msg) {
        s.sendSuccess(() -> Component.literal("[pvpbot] " + msg), false);
    }

    public static boolean debugOn() {
        return debug;
    }

    public static ArenaManager arena() {
        return arena;
    }
}
