package com.example.gaitrehabapp.models;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;
import androidx.annotation.NonNull;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class ModelPredictor {
    private static final String TAG = "ModelPredictor";
    private final OrtEnvironment env;
    private final OrtSession session;
    private final float[] mean;
    private final float[] scale;
    private final String inputName;
    private final List<String> outputNames;

    public ModelPredictor(Context context) throws Exception {
        env = OrtEnvironment.getEnvironment();

        AssetManager assets = context.getAssets();

        // Load ONNX model bytes (don’t use InputStream.available())
        byte[] modelBytes = readAllBytes(assets.open("xgboost_moderate_conf.onnx"));
        session = env.createSession(modelBytes, new OrtSession.SessionOptions());

        // Resolve input + output names from the session
        Map<String, NodeInfo> inputs = session.getInputInfo();
        if (inputs.isEmpty()) throw new IllegalStateException("ONNX model has no inputs");
        inputName = inputs.keySet().iterator().next();
        Log.i(TAG, "Resolved ONNX input name: " + inputName);

        Map<String, NodeInfo> outputs = session.getOutputInfo();
        outputNames = new ArrayList<>(outputs.keySet());
        Log.i(TAG, "Model outputs: " + outputNames);

        // Load scaler arrays
        mean  = loadNpyArray(assets, "scaler_mean.npy");
        scale = loadNpyArray(assets, "scaler_scale.npy");
        Log.i(TAG, "Scaler loaded: mean len=" + mean.length + ", scale len=" + scale.length);
    }

    public int predict(float[] rawFeatures) throws OrtException {
        if (rawFeatures == null) throw new IllegalArgumentException("features null");

        int n = Math.min(rawFeatures.length, Math.min(mean.length, scale.length));
        if (n != rawFeatures.length) {
            Log.w(TAG, "Feature length (" + rawFeatures.length + ") != scaler length (mean=" +
                    mean.length + ", scale=" + scale.length + "). Standardizing first " + n + " features.");
        }

        // Standardize (copy so we don't mutate caller's array)
        float[] standardized = Arrays.copyOf(rawFeatures, rawFeatures.length);
        for (int i = 0; i < n; i++) {
            standardized[i] = (rawFeatures[i] - mean[i]) / scale[i];
        }

        try (OnnxTensor tensor = OnnxTensor.createTensor(env, new float[][]{standardized});
             OrtSession.Result result = session.run(Collections.singletonMap(inputName, tensor))) {
            logResultOutputs(result);
            Integer fromProbs = tryGetPredictionFromProbabilities(result);
            if (fromProbs != null) return fromProbs;
            Integer fromLabel = tryGetPredictionFromLabel(result);
            if (fromLabel != null) return fromLabel;

            throw new OrtException("Could not parse ONNX outputs into a prediction.");
        }
    }

    private Integer tryGetPredictionFromProbabilities(OrtSession.Result result) throws OrtException {
        OnnxValue v = getAnyByNames(result, "output_probability", "probabilities", "probability", "output_probabilities");
        if (v == null) return null;

        Object val = v.getValue();
        if (val instanceof float[][]) {
            float[][] arr = (float[][]) val;
            if (arr.length > 0) return argmax(arr[0]);
        } else if (val instanceof float[]) {
            float[] arr = (float[]) val;
            return argmax(arr);
        }
        return null;
    }

    private Integer tryGetPredictionFromLabel(OrtSession.Result result) throws OrtException {
        OnnxValue v = getAnyByNames(result, "output_label", "label", "classLabel", "output_class");
        if (v == null) return null;

        Object val = v.getValue();
        if (val instanceof long[]) {
            long[] arr = (long[]) val;
            return (int) (arr.length > 0 ? arr[0] : 0);
        } else if (val instanceof long[][]) {
            long[][] arr = (long[][]) val;
            if (arr.length > 0 && arr[0].length > 0) return (int) arr[0][0];
        } else if (val instanceof int[]) {
            int[] arr = (int[]) val;
            return arr.length > 0 ? arr[0] : 0;
        } else if (val instanceof Long) {
            return ((Long) val).intValue();
        }
        return null;
    }

    private OnnxValue getAnyByNames(OrtSession.Result result, String... names) throws OrtException {
        for (String wanted : names) {
            for (String out : outputNames) {
                if (out.equals(wanted)) {
                    Optional<OnnxValue> opt = result.get(out);
                    if (opt != null && opt.isPresent()) return opt.get();
                }
            }
        }

        for (String out : outputNames) {
            Optional<OnnxValue> opt = result.get(out);
            if (opt.isEmpty()) continue;

            OnnxValue v = opt.get();
            Object val = v.getValue();
            for (String wanted : names) {
                boolean wantsProb = wanted.toLowerCase().contains("prob");
                if (wantsProb) {
                    if (val instanceof float[] || val instanceof float[][]) return v;
                } else {
                    if (val instanceof long[] || val instanceof long[][] ||
                            val instanceof int[]  || val instanceof Long) return v;
                }
            }
        }
        return null;
    }

    private static int argmax(float[] arr) {
        int idx = 0;
        float best = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > best) { best = arr[i]; idx = i; }
        }
        return idx;
    }

    private static byte[] readAllBytes(InputStream is) throws IOException {
        try (InputStream in = is; ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            byte[] buf = new byte[8192];
            int r;
            while ((r = in.read(buf)) != -1) bos.write(buf, 0, r);
            return bos.toByteArray();
        }
    }

    private float[] loadNpyArray(AssetManager assets, String filename) throws IOException {
        try (InputStream is = assets.open(filename)) {
            return new NpyReader().readFloatArray(is);
        }
    }


    private void logResultOutputs(OrtSession.Result result) {
        try {
            for (String out : outputNames) {
                Optional<OnnxValue> opt = result.get(out);
                if (opt.isEmpty()) continue;

                OnnxValue v = opt.get();
                Object val = v.getValue();
                String type = (val == null) ? "null" : val.getClass().getSimpleName();
                String shape = getString(val);
                Log.d(TAG, "Output '" + out + "': type=" + type + " shape=" + shape);
            }
        } catch (Exception e) {
            Log.w(TAG, "logResultOutputs error: " + e.getMessage());
        }
    }

    @NonNull
    private static String getString(Object val) {
        String shape = "";
        if (val instanceof float[][] a) {
            shape = "[" + a.length + "][" + (a.length > 0 ? a[0].length : 0) + "]";
        } else if (val instanceof long[][] a) {
            shape = "[" + a.length + "][" + (a.length > 0 ? a[0].length : 0) + "]";
        } else if (val instanceof float[] a) {
            shape = "[" + a.length + "]";
        } else if (val instanceof long[] a) {
            shape = "[" + a.length + "]";
        } else if (val instanceof int[] a) {
            shape = "[" + a.length + "]";
        }
        return shape;
    }
}
