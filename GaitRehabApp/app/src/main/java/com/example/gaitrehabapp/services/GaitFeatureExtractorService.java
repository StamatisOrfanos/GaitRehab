package com.example.gaitrehabapp.services;

import com.example.gaitrehabapp.models.DataPoint;
import com.example.gaitrehabapp.models.GaitCycle;
import com.example.gaitrehabapp.models.GaitWindowResult;

import java.util.ArrayList;
import java.util.List;

/**
 * Real-time gait feature extraction matching the training pipeline:
 *  - Butterworth low-pass (fc=6Hz, order=2) with forward-backward (zero-phase)
 *  - Detrend the window (remove DC) before zero-crossings
 *  - Zero-crossings on filtered Z with small deadband
 *  - Motion / duration sanity checks to drop bad windows
 *  - Mean stance/swing across valid cycles in the window
 */
public class GaitFeatureExtractorService {
    private static final float FS_HZ = 100f;           // sample rate
    private static final float FC_HZ = 6f;             // low-pass cutoff
    private static final float DEADBAND = 2f;          // deg/s, ignore very small near-zero wiggles
    private static final float MIN_SEG_SEC = 0.08f;    // reject ultra-short stance/swing (<80 ms)
    private static final float MAX_SEG_SEC = 1.20f;    // reject overlong stance/swing (>1.2 s)
    private static final float MIN_P2P = 15f;          // deg/s, minimum peak-to-peak motion to accept a window
    private static final float MIN_CYCLE_SEC = 0.60f;  // min stride (stance+swing)
    private static final float MAX_CYCLE_SEC = 2.20f;  // max stride window

    public static GaitWindowResult featureExtraction(List<DataPoint> leftZ, List<DataPoint> rightZ) {
        // 1) Filter (zero-phase) both legs
        List<DataPoint> leftF  = butterworthFwdBack(leftZ, FC_HZ, FS_HZ);
        List<DataPoint> rightF = butterworthFwdBack(rightZ, FC_HZ, FS_HZ);

        // 2) Detrend by removing per-window mean to restore sign crossings if there's DC offset
        leftF  = detrendMean(leftF);
        rightF = detrendMean(rightF);

        // 3) Quick motion sanity check to avoid NaNs spam at rest
        if (!hasEnoughMotion(leftF) || !hasEnoughMotion(rightF)) {
            return new GaitWindowResult(Float.NaN, Float.NaN, Float.NaN, Float.NaN);
        }

        // 4) Detect cycles (zero-crossings with deadband + duration sanity)
        List<GaitCycle> leftCycles  = detectStanceSwing(leftF);
        List<GaitCycle> rightCycles = detectStanceSwing(rightF);

        float leftStance  = (float) meanStance(leftCycles);
        float leftSwing   = (float) meanSwing(leftCycles);
        float rightStance = (float) meanStance(rightCycles);
        float rightSwing  = (float) meanSwing(rightCycles);

        return new GaitWindowResult(leftStance, leftSwing, rightStance, rightSwing);
    }

    // ----------------- Cycle detection -----------------
    private static List<GaitCycle> detectStanceSwing(List<DataPoint> data) {
        List<GaitCycle> cycles = new ArrayList<>();
        if (data == null || data.size() < 3) return cycles;

        // Zero-crossings with deadband
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

        // Between consecutive zero-crossings, stance ends at the valley (min Z)
        for (int i = 0; i < zc.size() - 1; i++) {
            int start = zc.get(i);
            int end   = zc.get(i + 1);
            if (end - start <= 1) continue;

            // Find local minimum in [start, end)
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
            float cycle  = stance + swing;

            // Segment sanity checks
            if (stance < MIN_SEG_SEC || stance > MAX_SEG_SEC) continue;
            if (swing  < MIN_SEG_SEC || swing  > MAX_SEG_SEC) continue;
            if (cycle  < MIN_CYCLE_SEC || cycle > MAX_CYCLE_SEC) continue;

            cycles.add(new GaitCycle(stance, swing));
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

    private static boolean hasEnoughMotion(List<DataPoint> w) {
        if (w == null || w.size() < 3) return false;
        float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
        for (DataPoint d : w) { float v = d.z; if (v < min) min = v; if (v > max) max = v; }
        return (max - min) >= MIN_P2P;
    }

    private static List<DataPoint> detrendMean(List<DataPoint> in) {
        if (in == null || in.isEmpty()) return in;
        float mean = 0f;
        for (DataPoint d : in) mean += d.z;
        mean /= in.size();
        List<DataPoint> out = new ArrayList<>(in.size());
        for (DataPoint d : in) out.add(new DataPoint(d.z - mean, d.timestamp));
        return out;
    }

    // ----------------- Filtering (Butterworth 2nd order, zero-phase via forward-backward) -----------------
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
        // RBJ cookbook coefficients for 2nd-order low-pass
        double w0 = 2.0 * Math.PI * fc / fs;
        double cosw0 = Math.cos(w0), sinw0 = Math.sin(w0);
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
    }

    private static List<DataPoint> butterworthFwdBack(List<DataPoint> in, float fc, float fs) {
        if (in == null || in.size() < 3) return in;

        // Extract arrays and (optionally) remove duplicate/unsorted timestamps
        int n = in.size();
        float[] x = new float[n];
        long[] t = new long[n];
        for (int i = 0; i < n; i++) { x[i] = in.get(i).z; t[i] = in.get(i).timestamp; }

        // Forward filter
        Biquad lp = designButterLP(fs, fc);
        float[] y = lp.filter(x);
        // Backward filter (zero phase)
        reverseInPlace(y);
        y = lp.filter(y);
        reverseInPlace(y);

        // Pack back
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
