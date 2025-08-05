package com.example.gaitrehabapp.services;

import com.example.gaitrehabapp.models.DataPoint;
import com.example.gaitrehabapp.models.GaitCycle;
import com.example.gaitrehabapp.models.GaitWindowResult;

import java.util.ArrayList;
import java.util.List;

public class GaitFeatureExtractorService {

    public static GaitWindowResult extractFromBuffers(List<DataPoint> leftZ, List<DataPoint> rightZ) {
    GaitWindowResult result = new GaitWindowResult();

    List<GaitCycle> leftPhases = detectStanceSwing(leftZ);
    List<GaitCycle> rightPhases = detectStanceSwing(rightZ);

    result.leftStance = meanStance(leftPhases);
    result.leftSwing = meanSwing(leftPhases);
    result.rightStance = meanStance(rightPhases);
    result.rightSwing = meanSwing(rightPhases);

    return result;
}

    private static List<GaitCycle> detectStanceSwing(List<DataPoint> data) {
        List<GaitCycle> result = new ArrayList<>();
        if (data.size() < 3) return result;

        List<Integer> zeroCrossings = new ArrayList<>();
        for (int i = 1; i < data.size(); i++) {
            if ((data.get(i - 1).z > 0 && data.get(i).z < 0) || (data.get(i - 1).z < 0 && data.get(i).z > 0)) {
                zeroCrossings.add(i);
            }
        }

        for (int i = 0; i < zeroCrossings.size() - 1; i++) {
            int start = zeroCrossings.get(i);
            int end = zeroCrossings.get(i + 1);
            if (end - start <= 1) continue;

            int minIndex = start;
            float minVal = data.get(start).z;
            for (int j = start; j < end; j++) {
                if (data.get(j).z < minVal) {
                    minVal = data.get(j).z;
                    minIndex = j;
                }
            }

            long tStart = data.get(start).timestamp;
            long tMin = data.get(minIndex).timestamp;
            long tEnd = data.get(end).timestamp;

            double stance = (tMin - tStart) / 1000.0;
            double swing = (tEnd - tMin) / 1000.0;
            result.add(new GaitCycle(stance, swing));
        }

        return result;
    }

    private static double meanStance(List<GaitCycle> phases) {
        return phases.stream().mapToDouble(p -> p.stanceTime).average().orElse(Double.NaN);
    }

    private static double meanSwing(List<GaitCycle> phases) {
        return phases.stream().mapToDouble(p -> p.swingTime).average().orElse(Double.NaN);
    }
}
