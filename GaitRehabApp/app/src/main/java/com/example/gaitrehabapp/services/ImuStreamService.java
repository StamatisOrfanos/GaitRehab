package com.example.gaitrehabapp.services;

import static com.example.gaitrehabapp.services.GaitFeatureExtractorService.extractFromBuffers;
import android.app.Service;
import android.content.Intent;
import android.os.Binder;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Environment;
import android.util.Log;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import com.example.gaitrehabapp.models.CircularBuffer;
import com.example.gaitrehabapp.models.DataPoint;
import com.example.gaitrehabapp.models.GaitWindowResult;
import com.example.gaitrehabapp.models.ImuDevice;
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
    private static final int BUFFER_CAPACITY = 200;
    private static final int ANALYSIS_INTERVAL_MS = 2000;

    private final IBinder binder = new LocalBinder();
    private final Map<String, CircularBuffer> bufferMap = new HashMap<>();
    private final Map<String, Boolean> pausedMap = new HashMap<>();
    private final Map<String, StringBuilder> sessionBuffers = new HashMap<>();
    private final Handler analysisHandler = new Handler();
    private final Map<String, String> deviceToSideMap = new HashMap<>();

    private final List<DataPoint> leftZ = new ArrayList<>();
    private final List<DataPoint> rightZ = new ArrayList<>();

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

    private final Runnable analysisRunnable = new Runnable() {
        @Override
        public void run() {
            leftZ.clear();
            rightZ.clear();

            for (Map.Entry<String, CircularBuffer> entry : bufferMap.entrySet()) {
                String deviceId = entry.getKey();
                CircularBuffer buffer = entry.getValue();

                float[] zVals = buffer.getZArray();
                long[] timestamps = buffer.getTimestampArray();

                List<DataPoint> points = new ArrayList<>();
                for (int i = 0; i < zVals.length; i++) {
                    points.add(new DataPoint(zVals[i], timestamps[i]));
                }

                if ("left".equals(deviceToSideMap.get(deviceId))) {
                    leftZ.addAll(points);
                } else if ("right".equals(deviceToSideMap.get(deviceId))) {
                    rightZ.addAll(points);
                }
            }

            if (!leftZ.isEmpty() && !rightZ.isEmpty()) {
                GaitWindowResult result = extractFromBuffers(leftZ, rightZ);

                Log.i(TAG, "Left Stance:  " + result.leftStance + "s");
                Log.i(TAG, "Left Swing :  " + result.leftSwing + "s");
                Log.i(TAG, "Right Stance: " + result.rightStance + "s");
                Log.i(TAG, "Right Swing : " + result.rightSwing + "s");
            }

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

            if (!analysisHandler.hasCallbacks(analysisRunnable)) {
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
            // Stop sensors
            device.getBoard().getModule(Gyro.class).angularVelocity().stop();
            device.getBoard().getModule(Gyro.class).stop();
            Log.i(TAG, "Stopped streaming for " + device.getModel());
        } catch (Exception e) {
            Log.e(TAG, "Error stopping: " + e.getMessage());
        }

        // Export to CSV
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
}
