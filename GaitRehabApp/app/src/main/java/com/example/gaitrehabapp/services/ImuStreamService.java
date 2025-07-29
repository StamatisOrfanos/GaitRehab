package com.example.gaitrehabapp.services;

import android.content.Intent;
import android.os.Binder;
import android.os.IBinder;
import android.util.Log;

import androidx.annotation.Nullable;

import com.example.gaitrehabapp.Models.ImuDevice;
import com.mbientlab.metawear.module.Accelerometer;
import com.mbientlab.metawear.module.Gyro;

public class ImuStreamService extends android.app.Service {

    private static final String TAG = "IMU_STREAM";
    private static final float SAMPLING_FREQUENCY = 100;
    private static final float GS = 4;

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

    public void startStreaming(ImuDevice device) {
        if (device == null || device.getBoard() == null) {
            Log.e(TAG, "Device or board is null");
            return;
        }

        // Accelerometer setup
        Accelerometer acc = device.getBoard().getModule(Accelerometer.class);
        acc.configure()
                .odr(SAMPLING_FREQUENCY)
                .range(GS)
                .commit();

        acc.acceleration().addRouteAsync(source ->
                source.stream((data, env) -> {
                    Log.i(TAG, device.getName() + " | Acc: " + data.value(Accelerometer.AccelerationDataProducer.class));
                })
        ).continueWithTask(task -> {
            acc.acceleration().start();
            acc.start();
            return null;
        });

        // Gyroscope setup
        Gyro gyro = device.getBoard().getModule(Gyro.class);
//        gyro.configure()
//                .odr(SAMPLING_FREQUENCY)
//                .commit();

        gyro.angularVelocity().addRouteAsync(source ->
                source.stream((data, env) -> {
                    Log.i(TAG, device.getName() + " | Gyro: " + data.value(Gyro.AngularVelocityDataProducer.class));
                })
        ).continueWithTask(task -> {
            gyro.angularVelocity().start();
            gyro.start();
            return null;
        });
    }

    public void stopStreaming(ImuDevice device) {
        if (device == null || device.getBoard() == null) return;

        try {
            Accelerometer acc = device.getBoard().getModule(Accelerometer.class);
            acc.acceleration().stop();
            acc.stop();

            Gyro gyro = device.getBoard().getModule(Gyro.class);
            gyro.angularVelocity().stop();
            gyro.stop();

            Log.i(TAG, "Stopped streaming from " + device.getName());

        } catch (Exception e) {
            Log.e(TAG, "Failed to stop streaming: " + e.getMessage());
        }
    }
}