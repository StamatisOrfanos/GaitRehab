package com.example.gaitrehabapp.models;

import android.content.Context;
import android.content.res.AssetManager;
import android.os.Build;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class ModelPredictor {
    private static final String TAG = "ModelPredictor";
    private static final String MODEL_FILE = "xgboost_moderate_conf.onnx";
    private static final String MEAN_FILE  = "scaler_mean.npy";
    private static final String SCALE_FILE = "scaler_scale.npy";
    private final OrtEnvironment env;
    private final OrtSession session;
    private final float[] mean;
    private final float[] scale;
    private final String inputName;
    private final List<String> outputNames;

    @RequiresApi(api = Build.VERSION_CODES.TIRAMISU)
    public ModelPredictor(Context context) throws Exception {
        Log.i(TAG, "ModelPredictor() starting…");
        env = OrtEnvironment.getEnvironment();
        AssetManager assets = context.getAssets();

        // 1) List assets (helps confirm packaging)
        try {
            String[] root = assets.list("");
            Log.i(TAG, "Assets root: " + Arrays.toString(root));
        } catch (IOException e) {
            Log.w(TAG, "Could not list assets", e);
        }

        // 2) Load model bytes
        byte[] modelBytes;
        try (InputStream is = assets.open(MODEL_FILE)) {
            modelBytes = readAllBytes(is);
        }
        Log.i(TAG, "ONNX model bytes: " + modelBytes.length);

        // 3) Create session
        session = env.createSession(modelBytes, new OrtSession.SessionOptions());

        // 4) Input/Output names
        Map<String, NodeInfo> inputs = session.getInputInfo();
        if (inputs.isEmpty()) throw new IllegalStateException("ONNX model has no inputs");
        inputName = inputs.keySet().iterator().next();
        Log.i(TAG, "Input name: " + inputName);

        Map<String, NodeInfo> outputs = session.getOutputInfo();
        outputNames = new ArrayList<>(outputs.keySet());
        Log.i(TAG, "Outputs: " + outputNames);

        // 5) Load scalers (accept float32 or float64)
        mean  = loadNpyAsFloatArray(assets, MEAN_FILE);
        scale = loadNpyAsFloatArray(assets, SCALE_FILE);
        Log.i(TAG, "Scaler loaded: mean len=" + mean.length + ", scale len=" + scale.length);
        if (mean.length == 0 || scale.length == 0) {
            throw new IllegalStateException("Scaler arrays empty");
        }
    }

    public int predict(float[] rawFeatures) throws OrtException {
        if (rawFeatures == null) throw new IllegalArgumentException("features null");

        int n = Math.min(rawFeatures.length, Math.min(mean.length, scale.length));
        if (n != rawFeatures.length) {
            Log.w(TAG, "Features (" + rawFeatures.length + ") != scaler len (mean=" + mean.length + ", scale=" + scale.length + "). Using first " + n + " features.");
        }

        float[] standardized = Arrays.copyOf(rawFeatures, rawFeatures.length);
        for (int i = 0; i < n; i++) {
            float s = (scale[i] == 0f) ? 1f : scale[i];
            standardized[i] = (rawFeatures[i] - mean[i]) / s;
        }
        Log.d(TAG, "Standardized: " + Arrays.toString(standardized));

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


    private static byte[] readAllBytes(InputStream is) throws IOException {
        try (InputStream in = is; ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            byte[] buf = new byte[8192]; int r;
            while ((r = in.read(buf)) != -1) bos.write(buf, 0, r);
            return bos.toByteArray();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.TIRAMISU)
    private static float[] loadNpyAsFloatArray(AssetManager assets, String filename) throws IOException {
        try (InputStream is = assets.open(filename)) {
            return readNpy1DAsFloat(is);
        }
    }

    private static float[] readNpy1DAsFloat(InputStream inputStream) throws IOException {
        DataInputStream dis = new DataInputStream(inputStream);

        // magic: 0x93 'N' 'U' 'M' 'P' 'Y'
        byte[] magic = new byte[6];
        dis.readFully(magic);
        if ((magic[0] & 0xFF) != 0x93 ||
                magic[1] != 'N' || magic[2] != 'U' || magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
            throw new IOException("Not an NPY file (magic mismatch)");
        }

        // version
        int vMajor = dis.readUnsignedByte();
        int vMinor = dis.readUnsignedByte();

        // header length (v1 uses uint16 little-endian, v2+ uses uint32 little-endian)
        int headerLen;
        if (vMajor == 1) {
            // little-endian uint16
            int b0 = dis.readUnsignedByte();
            int b1 = dis.readUnsignedByte();
            headerLen = (b1 << 8) | b0;
        } else {
            // little-endian uint32
            int b0 = dis.readUnsignedByte();
            int b1 = dis.readUnsignedByte();
            int b2 = dis.readUnsignedByte();
            int b3 = dis.readUnsignedByte();
            headerLen = (b3 << 24) | (b2 << 16) | (b1 << 8) | b0;
        }

        byte[] header = new byte[headerLen];
        dis.readFully(header);
        String hdr = new String(header, java.nio.charset.StandardCharsets.US_ASCII).trim();

        boolean little = hdr.contains("'<f4'") || hdr.contains("\"<f4\"") || hdr.contains("'<f8'") || hdr.contains("\"<f8\"");
        if (!little) {
            throw new IOException("Only little-endian NPY supported; header=" + hdr);
        }

        boolean f4 = hdr.contains("'descr': '<f4'") || hdr.contains("\"descr\": \"<f4\"");
        boolean f8 = hdr.contains("'descr': '<f8'") || hdr.contains("\"descr\": \"<f8\"");
        if (!f4 && !f8) throw new IOException("Only float32/float64 supported; header=" + hdr);

        // parse shape (assume 1-D like (123,) )
        int idxOpen = hdr.indexOf('(');
        int idxClose = hdr.indexOf(')');
        if (idxOpen < 0 || idxClose < 0 || idxClose <= idxOpen) {
            throw new IOException("Could not parse shape from header: " + hdr);
        }
        String inside = hdr.substring(idxOpen + 1, idxClose).trim();
        String lenStr = inside.replace(",", "").trim();
        int n = Integer.parseInt(lenStr);

        // read remaining data
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int r;
        while ((r = dis.read(buf)) != -1) bos.write(buf, 0, r);
        byte[] data = bos.toByteArray();

        java.nio.ByteBuffer bb = java.nio.ByteBuffer.wrap(data).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[n];
        if (f4) {
            java.nio.FloatBuffer fb = bb.asFloatBuffer();
            if (fb.remaining() < n) throw new IOException("NPY data too short for f4 count=" + n);
            fb.get(out, 0, n);
        } else {
            java.nio.DoubleBuffer db = bb.asDoubleBuffer();
            if (db.remaining() < n) throw new IOException("NPY data too short for f8 count=" + n);
            for (int i = 0; i < n; i++) out[i] = (float) db.get();
        }
        return out;
    }


    private void logResultOutputs(OrtSession.Result result) {
        try {
            for (String out : outputNames) {
                var opt = result.get(out);
                if (opt.isEmpty()) continue;
                var v = opt.get();
                Object val = v.getValue();
                String type = (val == null) ? "null" : val.getClass().getSimpleName();
                Log.d(TAG, "Output '" + out + "': type=" + type + " shape=" + shapeOf(val));
            }
        } catch (Exception e) {
            Log.w(TAG, "logResultOutputs error", e);
        }
    }

    @NonNull private static String shapeOf(Object val) {
        if (val instanceof float[][] a) return "[" + a.length + "][" + (a.length > 0 ? a[0].length : 0) + "]";
        if (val instanceof long[][] a)  return "[" + a.length + "][" + (a.length > 0 ? a[0].length : 0) + "]";
        if (val instanceof float[] a)   return "[" + a.length + "]";
        if (val instanceof long[] a)    return "[" + a.length + "]";
        if (val instanceof int[] a)     return "[" + a.length + "]";
        return "(unknown)";
    }

    private Integer tryGetPredictionFromProbabilities(OrtSession.Result result) throws OrtException {
        OnnxValue v = getAnyByNames(result, "output_probability", "probabilities", "probability", "output_probabilities");
        if (v == null) return null;
        Object val = v.getValue();
        if (val instanceof float[][] a && a.length > 0) return argmax(a[0]);
        if (val instanceof float[] a) return argmax(a);
        return null;
    }

    private Integer tryGetPredictionFromLabel(OrtSession.Result result) throws OrtException {
        OnnxValue v = getAnyByNames(result, "output_label", "label", "classLabel", "output_class");
        if (v == null) return null;
        Object val = v.getValue();
        if (val instanceof long[] la)   return la.length > 0 ? (int) la[0] : 0;
        if (val instanceof long[][] la) return (la.length > 0 && la[0].length > 0) ? (int) la[0][0] : 0;
        if (val instanceof int[] ia)    return ia.length > 0 ? ia[0] : 0;
        if (val instanceof Long L)      return L.intValue();
        return null;
    }

    private OnnxValue getAnyByNames(OrtSession.Result result, String... names) throws OrtException {
        for (String wanted : names) {
            for (String out : outputNames) {
                if (out.equals(wanted)) {
                    var opt = result.get(out);
                    if (opt != null && opt.isPresent()) return opt.get();
                }
            }
        }
        for (String out : outputNames) {
            var opt = result.get(out);
            if (opt.isEmpty()) continue;
            var v = opt.get();
            Object val = v.getValue();
            for (String wanted : names) {
                boolean wantsProb = wanted.toLowerCase().contains("prob");
                if (wantsProb && (val instanceof float[] || val instanceof float[][])) return v;
                if (!wantsProb && (val instanceof long[] || val instanceof long[][] || val instanceof int[] || val instanceof Long)) return v;
            }
        }
        return null;
    }

    private static int argmax(float[] arr) {
        int idx = 0; float best = arr[0];
        for (int i = 1; i < arr.length; i++) if (arr[i] > best) { best = arr[i]; idx = i; }
        return idx;
    }
}
