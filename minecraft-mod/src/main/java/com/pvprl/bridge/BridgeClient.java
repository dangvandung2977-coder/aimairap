package com.pvprl.bridge;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;

/** Localhost TCP client for the Python inference server. Every failure mode
 *  returns null (safe fallback: neutral controls, camera held); the caller
 *  retries on a later tick, so reconnects are automatic. Nothing here ever
 *  throws into game logic. */
public class BridgeClient {
    private final String host;
    private final int port;
    private final int connectTimeoutMs;
    private final int readTimeoutMs;
    private long lastErrorMs;
    private String lastError = "none";

    public BridgeClient(String host, int port,
                        int connectTimeoutMs, int readTimeoutMs) {
        this.host = host;
        this.port = port;
        this.connectTimeoutMs = connectTimeoutMs;
        this.readTimeoutMs = readTimeoutMs;
    }

    /** Sends one observation, returns the raw response line, or null. */
    public String query(long tick, double[] obs) {
        Socket s = new Socket();
        try {
            s.connect(new InetSocketAddress(host, port), connectTimeoutMs);
            s.setSoTimeout(readTimeoutMs);
            PrintWriter out = new PrintWriter(new OutputStreamWriter(
                    s.getOutputStream(), StandardCharsets.UTF_8), true);
            BufferedReader in = new BufferedReader(new InputStreamReader(
                    s.getInputStream(), StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder("{\"version\":1,\"tick\":");
            sb.append(tick).append(",\"observation\":[");
            for (int i = 0; i < obs.length; i++) {
                if (i > 0) sb.append(',');
                double v = obs[i];
                sb.append(Double.isFinite(v) ? Double.toString(v) : "0.0");
            }
            sb.append("],\"metadata\":{}}");
            out.println(sb);
            String line = in.readLine();
            s.close();
            if (line == null || line.isEmpty()) {
                noteError("empty_response");
                return null;
            }
            return line;
        } catch (Exception e) {
            noteError(e.getClass().getSimpleName());
            try { s.close(); } catch (Exception ignored) {}
            return null;
        }
    }

    private void noteError(String e) {
        lastError = e;
        lastErrorMs = System.currentTimeMillis();
    }

    public String status() {
        long ago = System.currentTimeMillis() - lastErrorMs;
        return lastError.equals("none") ? "OK"
                : lastError + " (" + (ago / 1000) + "s ago)";
    }
}
