package com.example.gaitrehabapp.services;

import android.app.Service;
import android.content.Intent;
import android.os.Binder;
import android.os.Build;
import android.os.IBinder;
import android.os.Environment;
import android.util.Log;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import com.example.gaitrehabapp.models.ImuDevice;
import com.mbientlab.metawear.data.AngularVelocity;
import com.mbientlab.metawear.module.Accelerometer;
import com.mbientlab.metawear.module.Gyro;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


public class ImuStreamService extends Service {

    private static final String TAG = "IMU_STREAM";

    private final IBinder binder = new LocalBinder();
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

    private final Map<String, Boolean> pausedMap = new HashMap<>();
    private final Map<String, StringBuilder> sessionBuffers = new HashMap<>();

    public void startStreaming(ImuDevice device, GyroZCallback zCallback) {
        if (device == null || device.getBoard() == null) {
            Log.e(TAG, "Cannot stream: null device or board");
            return;
        }

        String deviceId = device.getMacAddress();
        pausedMap.put(deviceId, false);
        sessionBuffers.put(deviceId, new StringBuilder());
        StringBuilder buffer = sessionBuffers.get(deviceId);

        // Gyroscope
        Gyro gyro = device.getBoard().getModule(Gyro.class);
        gyro.configure().odr(Gyro.OutputDataRate.ODR_50_HZ).commit();

        gyro.angularVelocity().addRouteAsync(source ->
                source.stream((data, env) -> {
                    if (Boolean.FALSE.equals(pausedMap.get(deviceId))) {
                        AngularVelocity gyroData = data.value(AngularVelocity.class);
                        long timestamp = System.currentTimeMillis();
                        buffer.append(timestamp).append(",gyro,")
                                .append(gyroData.x()).append(",")
                                .append(gyroData.y()).append(",")
                                .append(gyroData.z()).append("\n");
                        zCallback.onGyroZ(gyroData.z());
                    }
                })
        ).continueWithTask(task -> {
            gyro.angularVelocity().start();
            gyro.start();
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
            device.getBoard().getModule(Accelerometer.class).acceleration().stop();
            device.getBoard().getModule(Accelerometer.class).stop();

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
