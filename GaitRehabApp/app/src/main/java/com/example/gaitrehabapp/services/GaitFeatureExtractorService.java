package com.example.gaitrehabapp.services;

import com.example.gaitrehabapp.models.DataPoint;
import com.example.gaitrehabapp.models.GaitCycle;
import com.example.gaitrehabapp.models.GaitWindowResult;

import java.util.ArrayList;
import java.util.List;

public class GaitFeatureExtractorService {

    private static final float FS_HZ = 100f;
    private static final float FC_HZ = 6f;
    private static final float BASE_DEADBAND_HI = 2f;   // deg/s
    private static final float RMS_TO_HI = 0.25f;       // hi ~= 0.35 * RMS
    private static final float HI_TO_LO = 0.5f;         // lo = 0.5 * hi
    private static final float MIN_P2P = 8f;           // deg/s (min peak-to-peak to consider window)
    private static final float MIN_SEG_SEC = 0.08f;     // 80 ms
    private static final float MAX_SEG_SEC = 1.20f;
    private static final float MIN_CYCLE_SEC = 0.60f;
    private static final float MAX_CYCLE_SEC = 2.20f;

    public static GaitWindowResult featureExtraction(List<DataPoint> leftZ, List<DataPoint> rightZ) {
        List<DataPoint> lf = detrendMean(butterworthFwdBack(leftZ,  FC_HZ, FS_HZ));
        List<DataPoint> rf = detrendMean(butterworthFwdBack(rightZ, FC_HZ, FS_HZ));

        if (!hasEnoughMotion(lf) || !hasEnoughMotion(rf)) {
            return new GaitWindowResult(Float.NaN, Float.NaN, Float.NaN, Float.NaN);
        }

        // Try normal + flipped; pick the one with more valid cycles
        List<GaitCycle> lNorm = detectStanceSwing(lf);
        List<GaitCycle> lFlip = detectStanceSwing(flip(lf));
        List<GaitCycle> rNorm = detectStanceSwing(rf);
        List<GaitCycle> rFlip = detectStanceSwing(flip(rf));

        List<GaitCycle> lBest = (lFlip.size() > lNorm.size()) ? lFlip : lNorm;
        List<GaitCycle> rBest = (rFlip.size() > rNorm.size()) ? rFlip : rNorm;

        float lStance = (float) meanStance(lBest);
        float lSwing  = (float) meanSwing(lBest);
        float rStance = (float) meanStance(rBest);
        float rSwing  = (float) meanSwing(rBest);

        return new GaitWindowResult(lStance, lSwing, rStance, rSwing);
    }

    private static List<DataPoint> flip(List<DataPoint> in) {
        List<DataPoint> out = new ArrayList<>(in.size());
        for (DataPoint d : in) out.add(new DataPoint(-d.z, d.timestamp));
        return out;
    }

    private static List<GaitCycle> detectStanceSwing(List<DataPoint> w) {
        List<GaitCycle> out = new ArrayList<>();
        if (w == null || w.size() < 3) return out;

        // Adaptive hysteresis thresholds from window RMS
        float rms = rms(w);
        float hi  = Math.max(BASE_DEADBAND_HI, RMS_TO_HI * rms);
        float lo  = hi * HI_TO_LO;

        List<Integer> zc = schmittZeroCrossings(w, hi, lo);
        if (zc.size() < 2) return out;

        // Between consecutive ZCs: stance = start->min, swing = min->end
        for (int i = 0; i < zc.size() - 1; i++) {
            int start = zc.get(i);
            int end   = zc.get(i + 1);
            if (end - start <= 2) continue; // ultra-short

            // Find min strictly inside (avoid edges)
            int minIdx = -1;
            float minVal = Float.MAX_VALUE;
            for (int j = start + 1; j < end; j++) {
                float v = w.get(j).z;
                if (v < minVal) { minVal = v; minIdx = j; }
            }
            if (minIdx <= start || minIdx >= end) continue;

            long tStart = w.get(start).timestamp;
            long tMin   = w.get(minIdx).timestamp;
            long tEnd   = w.get(end).timestamp;

            float stance = (tMin - tStart) / 1000f;
            float swing  = (tEnd - tMin) / 1000f;
            float cycle  = stance + swing;

            if (stance < MIN_SEG_SEC || stance > MAX_SEG_SEC) continue;
            if (swing  < MIN_SEG_SEC || swing  > MAX_SEG_SEC) continue;
            if (cycle  < MIN_CYCLE_SEC || cycle > MAX_CYCLE_SEC) continue;

            out.add(new GaitCycle(stance, swing));
        }
        return out;
    }

    /**
     * Schmitt-trigger zero-crossings with hysteresis.
     * We only register a crossing when we were confidently on one side (>|hi|),
     * then the signal passes through the low band (<=|lo|) and emerges on the other side (>|hi|).
     */
    private static List<Integer> schmittZeroCrossings(List<DataPoint> w, float hi, float lo) {
        List<Integer> idx = new ArrayList<>();
        if (w.size() < 2) return idx;

        final int STATE_UNKNOWN = 0, STATE_POS = 1, STATE_NEG = -1;
        int state = STATE_UNKNOWN;

        for (int i = 1; i < w.size(); i++) {
            float prev = w.get(i - 1).z;
            float curr = w.get(i).z;

            // Update state with hysteresis bands
            if (state == STATE_UNKNOWN) {
                if (curr > hi)       state = STATE_POS;
                else if (curr < -hi) state = STATE_NEG;
            } else if (state == STATE_POS) {
                if (curr < lo && curr > -lo) {
                    // inside the neutral zone, keep waiting for a true cross
                } else if (curr < -hi) {
                    // crossed to negative side with certainty -> mark zero-crossing index
                    idx.add(i);
                    state = STATE_NEG;
                }
            } else {
                if (curr < lo && curr > -lo) {
                    // inside the neutral zone
                } else if (curr > hi) {
                    idx.add(i);
                    state = STATE_POS;
                }
            }
        }
        return idx;
    }

    private static boolean hasEnoughMotion(List<DataPoint> w) {
        if (w == null || w.size() < 3) return true;
        float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
        for (DataPoint d : w) { float v = d.z; if (v < min) min = v; if (v > max) max = v; }
        return !((max - min) >= MIN_P2P);
    }

    private static float rms(List<DataPoint> w) {
        double s2 = 0.0;
        for (DataPoint d : w) s2 += d.z * d.z;
        return (float) Math.sqrt(s2 / Math.max(1, w.size()));
    }

    private static List<DataPoint> detrendMean(List<DataPoint> in) {
        if (in == null || in.isEmpty()) return in;
        float mean = 0f; for (DataPoint d : in) mean += d.z; mean /= in.size();
        List<DataPoint> out = new ArrayList<>(in.size());
        for (DataPoint d : in) out.add(new DataPoint(d.z - mean, d.timestamp));
        return out;
    }

    private static double meanStance(List<GaitCycle> cs) {
        if (cs == null || cs.isEmpty()) return Double.NaN;
        double s = 0; for (GaitCycle c : cs) s += c.stanceTime; return s / cs.size();
    }
    private static double meanSwing(List<GaitCycle> cs) {
        if (cs == null || cs.isEmpty()) return Double.NaN;
        double s = 0; for (GaitCycle c : cs) s += c.swingTime; return s / cs.size();
    }

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

        return new Biquad((float)(b0/a0), (float)(b1/a0), (float)(b2/a0),
                (float)(a1/a0), (float)(a2/a0));
    }

    private static List<DataPoint> butterworthFwdBack(List<DataPoint> in, float fc, float fs) {
        if (in == null || in.size() < 3) return in;

        int n = in.size();
        float[] x = new float[n];
        long[]  t = new long[n];
        for (int i = 0; i < n; i++) { x[i] = in.get(i).z; t[i] = in.get(i).timestamp; }

        Biquad lp = designButterLP(fs, fc);

        float[] y = lp.filter(x);
        reverseInPlace(y);
        y = lp.filter(y);
        reverseInPlace(y);

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
