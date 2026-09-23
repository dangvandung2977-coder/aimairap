package com.pvprl.debug;

import java.util.concurrent.atomic.AtomicReference;

/** Last-bridge-call debug snapshot for /pvpbot debug + status. */
public class DebugState {
    private static final AtomicReference<String> LAST_RESPONSE = new AtomicReference<>("none");
    private static volatile long lastTick = -1;
    private static volatile double lastRoundTripMs = -1;

    public static void noteInference(String responseLine) {
        LAST_RESPONSE.set(responseLine.length() > 400
                ? responseLine.substring(0, 400) : responseLine);
    }

    public static void noteTick(long tick, double roundTripMs) {
        lastTick = tick;
        lastRoundTripMs = roundTripMs;
    }

    public static String summary() {
        return "tick=" + lastTick + " rtt=" + String.format("%.2f", lastRoundTripMs)
                + "ms resp=" + LAST_RESPONSE.get();
    }
}
