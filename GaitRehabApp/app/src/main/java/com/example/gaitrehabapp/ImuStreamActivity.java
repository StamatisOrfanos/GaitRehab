package com.example.gaitrehabapp;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Bundle;
import android.os.IBinder;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import com.example.gaitrehabapp.models.ImuDevice;
import com.example.gaitrehabapp.services.ImuStreamService;
import com.mbientlab.metawear.MetaWearBoard;
import com.mbientlab.metawear.android.BtleService;
import java.util.List;

public class ImuStreamActivity extends AppCompatActivity {

    private boolean isPaused = false;
    private TextView device1Name, device1GyroZ;
    private TextView device2Name, device2GyroZ;
    private List<ImuDevice> selectedDevices;

    private BtleService.LocalBinder btleBinder;
    private ImuStreamService streamService;

    private final ServiceConnection btleConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            btleBinder = (BtleService.LocalBinder) service;

            for (ImuDevice device : selectedDevices) {
                MetaWearBoard board = btleBinder.getMetaWearBoard(device.getBluetoothDevice());
                device.setBoard(board);
                board.connectAsync().continueWith(task -> {
                    if (!task.isFaulted()) {
                        device.setConnected(true);
                        runOnUiThread(() -> startStreamingForDevice(device));
                    } else {
                        Log.e("IMU", "Connection failed: " + device.getName());
                    }
                    return null;
                });
            }
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            btleBinder = null;
        }
    };

    private final ServiceConnection streamConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            streamService = ((ImuStreamService.LocalBinder) service).getService();
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            streamService = null;
        }
    };

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_imu_stream);
        device1Name = findViewById(R.id.device1Name);
        device1GyroZ = findViewById(R.id.device1GyroZ);
        device2Name = findViewById(R.id.device2Name);
        device2GyroZ = findViewById(R.id.device2GyroZ);

        selectedDevices = getIntent().getParcelableArrayListExtra("selected_devices");

        if (selectedDevices == null || selectedDevices.isEmpty()) finish();

        if (selectedDevices.size() == 1) {
            device2Name.setText("");
            device2GyroZ.setText("");
        }

        Button pauseResumeButton = findViewById(R.id.pauseResumeButton);

        pauseResumeButton.setOnClickListener(v -> {
            if (streamService == null || selectedDevices == null) return;

            if (isPaused) {
                for (ImuDevice device : selectedDevices) {
                    streamService.resumeStreaming(device);
                }
                isPaused = false;
                pauseResumeButton.setText("Pause Streaming");
            } else {
                for (ImuDevice device : selectedDevices) {
                    streamService.pauseStreaming(device);
                }
                isPaused = true;
                pauseResumeButton.setText("Resume Streaming");
            }
        });

        bindService(new Intent(this, BtleService.class), btleConnection, Context.BIND_AUTO_CREATE);
        bindService(new Intent(this, ImuStreamService.class), streamConnection, Context.BIND_AUTO_CREATE);
    }

    private void startStreamingForDevice(ImuDevice device) {
        String deviceName = device.getName() != null ? device.getName() : device.getMacAddress();

        if (streamService == null || device.getBoard() == null) return;

        streamService.startStreaming(
                device,
                accelZ -> runOnUiThread(() -> {
                    Log.d("IMU", "Accel Z for " + deviceName + ": " + accelZ);
                }),
                gyroZ -> runOnUiThread(() -> {
                    if (device == selectedDevices.get(0)) {
                        device1Name.setText(deviceName);
                        device1GyroZ.setText("Gyro Z: " + gyroZ);
                    } else if (selectedDevices.size() > 1 && device == selectedDevices.get(1)) {
                        device2Name.setText(deviceName);
                        device2GyroZ.setText("Gyro Z: " + gyroZ);
                    }
                })
        );
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        unbindService(btleConnection);
        unbindService(streamConnection);
    }
}