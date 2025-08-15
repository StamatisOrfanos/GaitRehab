package com.example.gaitrehabapp.services;

import static com.example.gaitrehabapp.services.GaitFeatureExtractorService.featureExtraction;
import android.app.Service;
import android.content.Intent;
import android.os.Binder;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.os.IBinder;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.os.VibratorManager;
import android.util.Log;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import com.example.gaitrehabapp.models.CircularBuffer;
import com.example.gaitrehabapp.models.DataPoint;
import com.example.gaitrehabapp.models.GaitWindowResult;
import com.example.gaitrehabapp.models.ImuDevice;
import com.example.gaitrehabapp.models.ModelPredictor;
import com.mbientlab.metawear.data.AngularVelocity;
import com.mbientlab.metawear.module.Gyro;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ImuStreamService extends Service {
    private static final String TAG = "IMU_STREAM";

    private static final int FS_HZ = 100;
    private static final int WINDOW_MS = 2000;
    private static final int WINDOW_SAMPLES = (FS_HZ * WINDOW_MS) / 1000;
    private static final int ANALYSIS_INTERVAL_MS = WINDOW_MS;
    private static final long INFERENCE_COOLDOWN_MS = WINDOW_MS;
    private static final long BUZZ_COOLDOWN_MS = WINDOW_MS;

    private static final int BUFFER_CAPACITY = WINDOW_SAMPLES;
    private long lastInferenceTs = 0L;
    private long lastBuzzTs = 0L;
    private boolean analysisStarted = false;

    private final IBinder binder = new LocalBinder();
    private final Map<String, CircularBuffer> bufferMap = new HashMap<>();
    private final Map<String, Boolean> pausedMap = new HashMap<>();
    private final Map<String, StringBuilder> sessionBuffers = new HashMap<>();
    private final Map<String, String> deviceToSideMap = new HashMap<>();
    private final Handler analysisHandler = new Handler();
    private final List<DataPoint> leftZ = new ArrayList<>();
    private final List<DataPoint> rightZ = new ArrayList<>();
    private ModelPredictor predictor;
    private Vibrator vibrator;

    public class LocalBinder extends Binder {
        public ImuStreamService getService() {
            return ImuStreamService.this;
        }
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        try {
            predictor = new ModelPredictor(getApplicationContext());
            Log.i(TAG, "ModelPredictor initialized successfully");
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize ModelPredictor: " + e.getMessage());
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            VibratorManager vm = (VibratorManager) getSystemService(VIBRATOR_MANAGER_SERVICE);
            vibrator = vm != null ? vm.getDefaultVibrator() : null;
        } else {
            vibrator = (Vibrator) getSystemService(VIBRATOR_SERVICE);
        }
    }

    private final Runnable analysisRunnable = new Runnable() {
        @Override
        public void run() {
            leftZ.clear();
            rightZ.clear();

            // Build 2s windows (last 200 samples) per device, mapped to left/right
            for (Map.Entry<String, CircularBuffer> entry : bufferMap.entrySet()) {
                String deviceId = entry.getKey();
                CircularBuffer buffer = entry.getValue();

                float[] zVals = buffer.getZArray();
                long[] timestamps = buffer.getTimestampArray();
                int len = zVals.length;
                if (len == 0) continue;

                int start = Math.max(0, len - WINDOW_SAMPLES);
                List<DataPoint> points = new ArrayList<>(len - start);
                for (int i = start; i < len; i++) {
                    points.add(new DataPoint(zVals[i], timestamps[i]));
                }

                String side = deviceToSideMap.get(deviceId);
                if ("left".equals(side)) {
                    leftZ.addAll(points);
                } else if ("right".equals(side)) {
                    rightZ.addAll(points);
                }
            }

            long now = System.currentTimeMillis();

            // Require a full window on both sides
            if (leftZ.size() < WINDOW_SAMPLES || rightZ.size() < WINDOW_SAMPLES) {
                analysisHandler.postDelayed(this, ANALYSIS_INTERVAL_MS);
                return;
            }

            // Throttle inferences to once per 2 seconds
            if (now - lastInferenceTs < INFERENCE_COOLDOWN_MS) {
                analysisHandler.postDelayed(this, ANALYSIS_INTERVAL_MS);
                return;
            }
            lastInferenceTs = now;

            // Extract features
            GaitWindowResult result = featureExtraction(leftZ, rightZ);

            Log.d(TAG, "==== Gait Values ====");
            Log.d(TAG, "Left Stance:  " + result.leftStance + "s");
            Log.d(TAG, "Left Swing :  " + result.leftSwing + "s");
            Log.d(TAG, "Right Stance: " + result.rightStance + "s");
            Log.d(TAG, "Right Swing : " + result.rightSwing + "s");

            float[] features = new float[] {
                    (float) result.leftStance, (float) result.leftSwing,
                    (float) result.rightStance, (float) result.rightSwing
            };

            try {
                if (predictor != null) {
                    int prediction = predictor.predict(features);
                    asymmetryAlert(prediction);
                    Log.d(TAG, "Predicted gait status: " + prediction);
                } else {
                    Log.w(TAG, "Predictor not initialized, skipping prediction");
                }
            } catch (Exception e) {
                Log.e(TAG, "Prediction failed: " + e.getMessage());
            }

            Log.d(TAG, "========================");

            analysisHandler.postDelayed(this, ANALYSIS_INTERVAL_MS);
        }
    };

    public void startStreaming(ImuDevice device, String side, GyroZCallback zCallback) {
        if (device == null || device.getBoard() == null) {
            Log.e(TAG, "Cannot stream: null device or board");
            return;
        }

        String deviceId = device.getMacAddress();
        pausedMap.put(deviceId, false);
        deviceToSideMap.put(deviceId, side.toLowerCase());

        Gyro gyro = device.getBoard().getModule(Gyro.class);
        gyro.configure().odr(Gyro.OutputDataRate.ODR_100_HZ).commit();

        gyro.angularVelocity().addRouteAsync(source ->
                source.stream((data, env) -> {
                    if (Boolean.FALSE.equals(pausedMap.get(deviceId))) {
                        AngularVelocity gyroData = data.value(AngularVelocity.class);
                        float z = gyroData.z();
                        long timestamp = System.currentTimeMillis();

                        CircularBuffer cb = bufferMap.computeIfAbsent(deviceId, id -> new CircularBuffer(BUFFER_CAPACITY));
                        cb.add(z, timestamp);

                        zCallback.onGyroZ(z);
                    }
                })
        ).continueWithTask(task -> {
            gyro.angularVelocity().start();
            gyro.start();

            if (!analysisStarted) {
                analysisStarted = true;
                analysisHandler.postDelayed(analysisRunnable, ANALYSIS_INTERVAL_MS);
            }

            return null;
        });
    }

    public void pauseStreaming(ImuDevice device) {
        if (device != null) {
            pausedMap.put(device.getMacAddress(), true);
            Log.i(TAG, "Paused streaming for " + device.getModel());
        }
    }

    public void resumeStreaming(ImuDevice device) {
        if (device != null) {
            pausedMap.put(device.getMacAddress(), false);
            Log.i(TAG, "Resumed streaming for " + device.getModel());
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.VANILLA_ICE_CREAM)
    public void stopStreaming(ImuDevice device) {
        if (device == null || device.getBoard() == null) return;

        String deviceId = device.getMacAddress();
        try {
            device.getBoard().getModule(Gyro.class).angularVelocity().stop();
            device.getBoard().getModule(Gyro.class).stop();
            Log.i(TAG, "Stopped streaming for " + device.getModel());
        } catch (Exception e) {
            Log.e(TAG, "Error stopping: " + e.getMessage());
        }

        exportToCSV(deviceId, sessionBuffers.get(deviceId));
        sessionBuffers.remove(deviceId);
        pausedMap.remove(deviceId);
    }

    @RequiresApi(api = Build.VERSION_CODES.VANILLA_ICE_CREAM)
    private void exportToCSV(String deviceId, StringBuilder buffer) {
        if (buffer == null || buffer.isEmpty()) return;

        File dir = new File(getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), "IMU_Logs");
        if (!dir.exists()) dir.mkdirs();

        File file = new File(dir, deviceId + "_session.csv");

        try (FileWriter writer = new FileWriter(file)) {
            writer.write("timestamp,type,x,y,z\n");
            writer.write(buffer.toString());
            Log.i(TAG, "Saved session to: " + file.getAbsolutePath());
        } catch (IOException e) {
            Log.e(TAG, "CSV export failed: " + e.getMessage());
        }
    }

    private void asymmetryAlert(int prediction) {
        if (prediction != 1 || vibrator == null || !vibrator.hasVibrator()) return;

        long now = System.currentTimeMillis();
        if (now - lastBuzzTs < BUZZ_COOLDOWN_MS) return;

        lastBuzzTs = now;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(300, VibrationEffect.DEFAULT_AMPLITUDE));
        } else {
            vibrator.vibrate(300);
        }
    }
}
