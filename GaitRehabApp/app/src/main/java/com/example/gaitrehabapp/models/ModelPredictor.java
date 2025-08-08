package com.example.gaitrehabapp.models;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class ModelPredictor {
    private final OrtEnvironment env;
    private final OrtSession session;
    private final float[] mean;
    private final float[] scale;

    public ModelPredictor(Context context) throws Exception {
        env = OrtEnvironment.getEnvironment();
        AssetManager assets = context.getAssets();

        InputStream modelStream = assets.open("xgboost_moderate_conf.onnx");
        byte[] modelBytes = new byte[modelStream.available()];
        modelStream.read(modelBytes);
        session = env.createSession(modelBytes, new OrtSession.SessionOptions());
        mean = loadNpyArray(assets, "scaler_mean.npy");
        scale = loadNpyArray(assets, "scaler_scale.npy");
    }

    private float[] loadNpyArray(AssetManager assets, String filename) throws IOException {
        try (InputStream is = assets.open(filename)) {
            return new NpyReader().readFloatArray(is);
        }
    }

    public int predict(float[] input) throws OrtException {
        float[] standardized = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            standardized[i] = (input[i] - mean[i]) / scale[i];
        }

        OnnxTensor tensor = OnnxTensor.createTensor(env, new float[][]{standardized});
        OrtSession.Result result = session.run(Collections.singletonMap("float_input", tensor));

        float[][] output = (float[][]) result.get(0).getValue();
        return Math.round(output[0][0]);
    }
}
