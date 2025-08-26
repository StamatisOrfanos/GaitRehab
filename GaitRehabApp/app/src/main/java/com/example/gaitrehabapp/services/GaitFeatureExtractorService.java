package com.example.gaitrehabapp.services;

import com.example.gaitrehabapp.models.DataPoint;
import com.example.gaitrehabapp.models.GaitCycle;
import com.example.gaitrehabapp.models.GaitWindowResult;
import java.util.ArrayList;
import java.util.List;

public class GaitFeatureExtractorService {
    private static final float b0 = 0.39133577f;
    private static final float b1 = 0.78267153f;
    private static final float b2 = 0.39133577f;
    private static final float a1 = -0.36952738f;
    private static final float a2 = 0.19581571f;

    // Cycle sanity limits (seconds)
    private static final float MIN_SEGMENT_S = 0.05f;
    private static final float MIN_CYCLE_S   = 0.30f;
    private static final float MAX_CYCLE_S   = 2.50f;

    public static GaitWindowResult featureExtraction(List<DataPoint> leftZ, List<DataPoint> rightZ) {
        float leftStance = Float.NaN, leftSwing = Float.NaN;
        float rightStance = Float.NaN, rightSwing = Float.NaN;

        if (leftZ != null && leftZ.size() >= 5) {
            float[] lf = filterZeroPhase(extractZ(leftZ));
            List<GaitCycle> lcycles = detectStanceSwing(lf, leftZ);
            leftStance = meanStance(lcycles);
            leftSwing  = meanSwing(lcycles);
        }

        if (rightZ != null && rightZ.size() >= 5) {
            float[] rf = filterZeroPhase(extractZ(rightZ));
            List<GaitCycle> rcycles = detectStanceSwing(rf, rightZ);
            rightStance = meanStance(rcycles);
            rightSwing  = meanSwing(rcycles);
        }
        return new GaitWindowResult(leftStance, leftSwing, rightStance, rightSwing);
    }

    // ---- Core steps ---------------------------------------------------------
    private static float[] extractZ(List<DataPoint> pts) {
        int n = pts.size();
        float[] z = new float[n];
        for (int i = 0; i < n; i++) z[i] = pts.get(i).z;
        return z;
    }

    private static float[] filterZeroPhase(float[] x) {
        float[] y = biquadFilter(x, b1, b2, a1, a2);
        reverseInPlace(y);
        y = biquadFilter(y, b1, b2, a1, a2);
        reverseInPlace(y);
        return y;
    }

    private static float[] biquadFilter(float[] x, float b1, float b2, float a1, float a2) {
        int n = x.length;
        float[] y = new float[n];

        float z1 = 0f, z2 = 0f; // state
        for (int i = 0; i < n; i++) {
            float w = x[i] - a1 * z1 - a2 * z2;
            float o = GaitFeatureExtractorService.b0 * w + b1 * z1 + b2 * z2;
            y[i] = o;
            z2 = z1;
            z1 = w;
        }
        return y;
    }

    private static void reverseInPlace(float[] a) {
        for (int i = 0, j = a.length - 1; i < j; i++, j--) {
            float t = a[i]; a[i] = a[j]; a[j] = t;
        }
    }

    private static int[] zeroCrossings(float[] zf) {
        List<Integer> idx = new ArrayList<>();
        for (int i = 1; i < zf.length; i++) {
            if ((zf[i - 1] > 0 && zf[i] <= 0) || (zf[i - 1] < 0 && zf[i] >= 0)) {
                idx.add(i);
            }
        }
        int[] out = new int[idx.size()];
        for (int i = 0; i < idx.size(); i++) out[i] = idx.get(i);
        return out;
    }

    private static List<GaitCycle> detectStanceSwing(float[] zFiltered, List<DataPoint> series) {
        List<GaitCycle> out = new ArrayList<>();
        if (zFiltered.length != series.size() || zFiltered.length < 3) return out;

        int[] zc = zeroCrossings(zFiltered);
        if (zc.length < 2) return out;

        for (int k = 0; k < zc.length - 1; k++) {
            int start = zc[k];
            int end   = zc[k + 1];
            if (end - start <= 1) continue;

            // find min z in [start, end)
            int minIdx = start;
            float minV = zFiltered[start];
            for (int i = start + 1; i < end; i++) {
                if (zFiltered[i] < minV) {
                    minV = zFiltered[i];
                    minIdx = i;
                }
            }

            long tStart = series.get(start).timestamp;
            long tMin   = series.get(minIdx).timestamp;
            long tEnd   = series.get(end).timestamp;

            float stance = (tMin - tStart) / 1000f;
            float swing  = (tEnd - tMin) / 1000f;
            float cycle  = (tEnd - tStart) / 1000f;

            // guards
            if (stance <= MIN_SEGMENT_S || swing <= MIN_SEGMENT_S) continue;
            if (cycle < MIN_CYCLE_S || cycle > MAX_CYCLE_S) continue;

            out.add(new GaitCycle(stance, swing));
        }
        return out;
    }

    private static float meanStance(List<GaitCycle> cycles) {
        if (cycles == null || cycles.isEmpty()) return Float.NaN;
        float s = 0f;
        for (GaitCycle c : cycles) s += (float) c.stanceTime;
        return s / cycles.size();
    }

    private static float meanSwing(List<GaitCycle> cycles) {
        if (cycles == null || cycles.isEmpty()) return Float.NaN;
        float s = 0f;
        for (GaitCycle c : cycles) s += (float) c.swingTime;
        return s / cycles.size();
    }

}
