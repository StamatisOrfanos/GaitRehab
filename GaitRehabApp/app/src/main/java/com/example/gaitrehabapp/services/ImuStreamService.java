package com.example.gaitrehabapp.services;

import static com.example.gaitrehabapp.services.GaitFeatureExtractorService.featureExtraction;
import android.app.Service;
import android.content.Intent;
import android.os.Binder;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.os.IBinder;
import android.os.SystemClock;
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
    private static final String TAG = "IMU_STREAM_SERVICE";
    private static final int FS_HZ = 100;
    private static final int WINDOW_MS = 2000;
    private static final int HOP_MS = 1000;
    private static final int WINDOW_SAMPLES = (FS_HZ * WINDOW_MS) / 1000;
    private static final int ANALYSIS_INTERVAL_MS = HOP_MS;
    private static final int BUFFER_CAPACITY = WINDOW_SAMPLES * 3;
    private static final long INFERENCE_COOLDOWN_MS = HOP_MS;
    private static final long BUZZ_COOLDOWN_MS = 2000L;
    private final IBinder binder = new LocalBinder();
    private final Map<String, CircularBuffer> bufferMap = new HashMap<>();
    private final Map<String, Boolean> pausedMap = new HashMap<>();
    private final Map<String, String> deviceToSideMap = new HashMap<>();
    private final Map<String, StringBuilder> sessionBuffers = new HashMap<>();
    private final Handler analysisHandler = new Handler();
    private final List<DataPoint> leftZ = new ArrayList<>();
    private final List<DataPoint> rightZ = new ArrayList<>();
    private boolean analysisStarted = false;
    private long lastInferenceTs = 0L;
    private long lastBuzzTs = 0L;
    private long lastProcessedEndTs = 0L;
    private ModelPredictor predictor;
    private Vibrator vibrator;
    public interface GyroZCallback { void onGyroZ(float z); }
    public class LocalBinder extends Binder {
        public ImuStreamService getService() { return ImuStreamService.this; }
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) { return binder; }

    @Override
    public void onCreate() {
        super.onCreate();
        try {
            Log.d(TAG, "Initializing ModelPredictor…");
            predictor = new ModelPredictor(getApplicationContext());
            Log.d(TAG, "ModelPredictor initialized successfully");
        } catch (Throwable t) {
            Log.e(TAG, "Failed to initialize ModelPredictor", t);
        }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            VibratorManager vm = (VibratorManager) getSystemService(VIBRATOR_MANAGER_SERVICE);
            vibrator = (vm != null) ? vm.getDefaultVibrator() : null;
        } else {
            vibrator = (Vibrator) getSystemService(VIBRATOR_SERVICE);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        analysisHandler.removeCallbacksAndMessages(null);
    }


    public void startStreaming(ImuDevice device, String side, GyroZCallback zCallback) {
        if (device == null || device.getBoard() == null) {
            Log.e(TAG, "Cannot stream: null device or board");
            return;
        }

        final String deviceId = device.getMacAddress();
        pausedMap.put(deviceId, false);
        deviceToSideMap.put(deviceId, side.toLowerCase());

        Gyro gyro = device.getBoard().getModule(Gyro.class);
        gyro.configure().odr(Gyro.OutputDataRate.ODR_100_HZ).commit();

        gyro.angularVelocity().addRouteAsync(source ->
                source.stream((data, env) -> {
                    if (Boolean.TRUE.equals(pausedMap.get(deviceId))) return;

                    AngularVelocity gyroscope = data.value(AngularVelocity.class);
                    float z = gyroscope.z();
                    long timestamp = System.currentTimeMillis();

                    CircularBuffer cb = bufferMap.computeIfAbsent(deviceId,
                            id -> new CircularBuffer(BUFFER_CAPACITY));
                    cb.add(z, timestamp);

                    if (zCallback != null) zCallback.onGyroZ(z);
                })
        ).continueWithTask(task -> {
            gyro.angularVelocity().start();
            gyro.start();

            if (!analysisStarted) {
                analysisStarted = true;
                Log.d(TAG, "Starting analysis loop");
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

    private final Runnable analysisRunnable = new Runnable() {
        @Override
        public void run() {
            leftZ.clear();
            rightZ.clear();

            // Collect all points for each side using a CONSISTENT snapshot per buffer
            List<DataPoint> allLeft = new ArrayList<>();
            List<DataPoint> allRight = new ArrayList<>();

            for (Map.Entry<String, CircularBuffer> entry : bufferMap.entrySet()) {
                final String deviceId = entry.getKey();
                final CircularBuffer buffer = entry.getValue();
                final String side = deviceToSideMap.get(deviceId);

                CircularBuffer.Snapshot snap = buffer.snapshot();
                float[] zVals = snap.z();
                long[]  tVals = snap.t();
                int len = zVals.length;
                if (len == 0) continue;

                List<DataPoint> points = new ArrayList<>(len);
                for (int i = 0; i < len; i++) {
                    points.add(new DataPoint(zVals[i], tVals[i]));
                }

                if ("left".equals(side))  allLeft.addAll(points);
                if ("right".equals(side)) allRight.addAll(points);
            }

            if (allLeft.isEmpty() || allRight.isEmpty()) {
                analysisHandler.postDelayed(this, ANALYSIS_INTERVAL_MS);
                return;
            }

            // Align both legs to the same window [endTs - WINDOW_MS, endTs]
            long latestLeftTs  = allLeft.get(allLeft.size() - 1).timestamp;
            long latestRightTs = allRight.get(allRight.size() - 1).timestamp;
            long endTs = Math.min(latestLeftTs, latestRightTs);
            long startTs = endTs - WINDOW_MS;

            // Ensure the window advances by at least the hop
            if (endTs - lastProcessedEndTs < HOP_MS) {
                analysisHandler.postDelayed(this, ANALYSIS_INTERVAL_MS);
                return;
            }

            // Slice by timestamp for a 2s window
            for (DataPoint p : allLeft)  if (p.timestamp >= startTs && p.timestamp <= endTs) leftZ.add(p);
            for (DataPoint p : allRight) if (p.timestamp >= startTs && p.timestamp <= endTs) rightZ.add(p);

            // Require most of the expected samples (allow some jitter)
            if (leftZ.size() < WINDOW_SAMPLES * 0.8f || rightZ.size() < WINDOW_SAMPLES * 0.8f) {
                analysisHandler.postDelayed(this, ANALYSIS_INTERVAL_MS);
                return;
            }

            long now = SystemClock.elapsedRealtime();
            if (now - lastInferenceTs < INFERENCE_COOLDOWN_MS) {
                analysisHandler.postDelayed(this, ANALYSIS_INTERVAL_MS);
                return;
            }
            lastInferenceTs = now;
            lastProcessedEndTs = endTs;

            // Get the features from the data and get the model prediction
            GaitWindowResult result = featureExtraction(leftZ, rightZ);
            modelPrediction(result);

            analysisHandler.postDelayed(this, ANALYSIS_INTERVAL_MS);
        }
    };


    private void modelPrediction(GaitWindowResult gaitWindowResult) {
        Log.d(TAG, "==== Gait Values ====");
        Log.d(TAG, "Left Stance:  " + gaitWindowResult.getLeftStance()  + "s");
        Log.d(TAG, "Left Swing :  " + gaitWindowResult.getLeftSwing()   + "s");
        Log.d(TAG, "Right Stance: " + gaitWindowResult.getRightStance() + "s");
        Log.d(TAG, "Right Swing : " + gaitWindowResult.getRightSwing()  + "s");


        if (!gaitWindowResult.gaitWindowValid() || predictor == null) {
            Log.w(TAG, "Predictor not initialized or invalid window, skipping prediction");
            Log.d(TAG, "========================");
            return;
        }

        // Model expects: RightStance, LeftStance, RightSwing, LeftSwing
        float[] modelInput = new float[] {
                gaitWindowResult.getRightStance(), gaitWindowResult.getLeftStance(),
                gaitWindowResult.getRightSwing(),  gaitWindowResult.getLeftSwing()
        };

        try {
            int prediction = predictor.predict(modelInput);
            Log.d(TAG, "Predicted gait status: " + prediction);
            if (prediction == 1 && windowIsFresh(lastProcessedEndTs)) buzzOnce();
        } catch (Exception e) {
            Log.e(TAG, "Prediction failed: " + e.getMessage());
        }

        Log.d(TAG, "========================");
    }

    private void buzzOnce() {
        if (vibrator == null || !vibrator.hasVibrator()) return;

        long now = SystemClock.elapsedRealtime();
        if (now - lastBuzzTs < BUZZ_COOLDOWN_MS) return;

        lastBuzzTs = now;
        try { vibrator.cancel(); } catch (Throwable ignored) {}
        vibrator.vibrate(VibrationEffect.createOneShot(120, VibrationEffect.DEFAULT_AMPLITUDE));
    }

    private boolean windowIsFresh(long endTs) {
        return (lastProcessedEndTs == endTs)
                && (leftZ.size() >= WINDOW_SAMPLES * 0.8f)
                && (rightZ.size() >= WINDOW_SAMPLES * 0.8f);
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

}