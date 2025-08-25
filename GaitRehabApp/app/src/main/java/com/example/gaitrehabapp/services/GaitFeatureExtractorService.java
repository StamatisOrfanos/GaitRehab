package com.example.gaitrehabapp.services;

import com.example.gaitrehabapp.models.DataPoint;
import com.example.gaitrehabapp.models.GaitCycle;
import com.example.gaitrehabapp.models.GaitWindowResult;

import java.util.ArrayList;
import java.util.List;

public class GaitFeatureExtractorService {
    private static final float FS_HZ = 100f;
    private static final float FC_HZ = 6f;
    private static final float DEADBAND = 5f;
    private static final float MIN_SEG_SEC = 0.08f;

    // ---- Public API: called by ImuStreamService ----
    public static GaitWindowResult featureExtraction(List<DataPoint> leftZ, List<DataPoint> rightZ) {
        // 1) Filter (zero-phase) both legs
        List<DataPoint> leftF  = butterworthFwdBack(leftZ, FC_HZ, FS_HZ);
        List<DataPoint> rightF = butterworthFwdBack(rightZ, FC_HZ, FS_HZ);

        // 2) Detect stance/swing cycles on filtered signals
        List<GaitCycle> leftCycles  = detectStanceSwing(leftF);
        List<GaitCycle> rightCycles = detectStanceSwing(rightF);

        // 3) Compute means
        float leftStance  = (float) meanStance(leftCycles);
        float leftSwing   = (float) meanSwing(leftCycles);
        float rightStance = (float) meanStance(rightCycles);
        float rightSwing  = (float) meanSwing(rightCycles);

        return new GaitWindowResult(leftStance, leftSwing, rightStance, rightSwing);
    }

    private static List<GaitCycle> detectStanceSwing(List<DataPoint> data) {
        List<GaitCycle> cycles = new ArrayList<>();
        if (data == null || data.size() < 3) return cycles;

        // Find zero crossings with a small deadband (both samples must be "active")
        List<Integer> zc = new ArrayList<>();
        for (int i = 1; i < data.size(); i++) {
            float p = data.get(i - 1).z, c = data.get(i).z;
            boolean pOn = Math.abs(p) >= DEADBAND;
            boolean cOn = Math.abs(c) >= DEADBAND;
            if (pOn && cOn && ((p > 0 && c < 0) || (p < 0 && c > 0))) {
                zc.add(i);
            }
        }
        if (zc.size() < 2) return cycles;

        // Between consecutive zero-crossings, stance ends at the minimum Z (valley)
        for (int i = 0; i < zc.size() - 1; i++) {
            int start = zc.get(i);
            int end   = zc.get(i + 1);
            if (end - start <= 1) continue;

            // Min index in [start, end)
            int minIdx = start;
            float minVal = data.get(start).z;
            for (int j = start + 1; j < end; j++) {
                if (data.get(j).z < minVal) {
                    minVal = data.get(j).z;
                    minIdx = j;
                }
            }

            long tStart = data.get(start).timestamp;
            long tMin   = data.get(minIdx).timestamp;
            long tEnd   = data.get(end).timestamp;

            float stance = (tMin - tStart) / 1000f;
            float swing  = (tEnd - tMin) / 1000f;

            // Reject ultra-short segments
            if (stance >= MIN_SEG_SEC && swing >= MIN_SEG_SEC) {
                cycles.add(new GaitCycle(stance, swing));
            }
        }

        return cycles;
    }

    private static double meanStance(List<GaitCycle> cycles) {
        if (cycles == null || cycles.isEmpty()) return Float.NaN;
        double s = 0;
        for (GaitCycle c : cycles) s += c.stanceTime;
        return s / cycles.size();
    }

    private static double meanSwing(List<GaitCycle> cycles) {
        if (cycles == null || cycles.isEmpty()) return Float.NaN;
        double s = 0;
        for (GaitCycle c : cycles) s += c.swingTime;
        return s / cycles.size();
    }

    // ---- Butterworth low-pass (order=2) + filtfilt (forward-backward) ----
    // RBJ cookbook biquad; for Butterworth 2nd-order: Q = 1/sqrt(2)
    private static class Biquad {
        final float b0, b1, b2, a1, a2;

        Biquad(float b0, float b1, float b2, float a1, float a2) {
            this.b0 = b0; this.b1 = b1; this.b2 = b2; this.a1 = a1; this.a2 = a2;
        }

        float[] filter(float[] x) {
            float[] y = new float[x.length];
            float z1 = 0f, z2 = 0f;
            for (int i = 0; i < x.length; i++) {
                float in = x[i];
                float out = in * b0 + z1;
                z1 = in * b1 + z2 - a1 * out;
                z2 = in * b2 - a2 * out;
                y[i] = out;
            }
            return y;
        }
    }

    private static Biquad designButterLP(float fs, float fc) {
        // RBJ: https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
        double w0 = 2.0 * Math.PI * fc / fs;
        double cosw0 = Math.cos(w0);
        double sinw0 = Math.sin(w0);
        double Q = Math.sqrt(0.5); // Butterworth
        double alpha = sinw0 / (2.0 * Q);

        double b0 = (1 - cosw0) / 2.0;
        double b1 = 1 - cosw0;
        double b2 = (1 - cosw0) / 2.0;
        double a0 = 1 + alpha;
        double a1 = -2 * cosw0;
        double a2 = 1 - alpha;

        float nb0 = (float) (b0 / a0);
        float nb1 = (float) (b1 / a0);
        float nb2 = (float) (b2 / a0);
        float na1 = (float) (a1 / a0);
        float na2 = (float) (a2 / a0);

        return new Biquad(nb0, nb1, nb2, na1, na2);
        // Note: This is a single biquad (order=2). filtfilt below achieves zero-phase like scipy.filtfilt(order=2).
    }

    private static List<DataPoint> butterworthFwdBack(List<DataPoint> in, float fc, float fs) {
        if (in == null || in.size() < 3) return in;

        // Extract z & timestamps
        int n = in.size();
        float[] x = new float[n];
        long[] t = new long[n];
        for (int i = 0; i < n; i++) {
            x[i] = in.get(i).z;
            t[i] = in.get(i).timestamp;
        }

        // Design LPF and apply forward
        Biquad lp = designButterLP(fs, fc);
        float[] y = lp.filter(x);

        // Reverse, filter again (backward), then reverse back => zero-phase
        reverseInPlace(y);
        y = lp.filter(y);
        reverseInPlace(y);

        // Repack
        List<DataPoint> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) out.add(new DataPoint(y[i], t[i]));
        return out;
    }

    private static void reverseInPlace(float[] a) {
        for (int i = 0, j = a.length - 1; i < j; i++, j--) {
            float tmp = a[i]; a[i] = a[j]; a[j] = tmp;
        }
    }
}
