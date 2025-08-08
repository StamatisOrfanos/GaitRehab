package com.example.gaitrehabapp;

import android.annotation.SuppressLint;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Bundle;
import android.os.IBinder;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import com.example.gaitrehabapp.models.ImuDevice;
import com.example.gaitrehabapp.services.ImuStreamService;
import com.mbientlab.metawear.MetaWearBoard;
import com.mbientlab.metawear.android.BtleService;
import java.util.List;

public class ImuStreamActivity extends AppCompatActivity {
    private boolean isPaused = false;
    private boolean btleReady = false;
    private boolean streamReady = false;
    private TextView device1Name, device1GyroZ;
    private TextView device2Name, device2GyroZ;
    private List<ImuDevice> selectedDevices;

    private BtleService.LocalBinder btleBinder;
    private ImuStreamService streamService;

    private final ServiceConnection btleConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            btleBinder = (BtleService.LocalBinder) service;
            btleReady = true;
            checkAndStartStreaming();
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
            streamReady = true;
            checkAndStartStreaming();
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

    private void checkAndStartStreaming() {
        if (!btleReady || !streamReady || selectedDevices == null) return;

        if (selectedDevices.size() < 2) {
            Toast.makeText(this, "Two IMU devices required for gait symmetry analysis.", Toast.LENGTH_LONG).show();
            return;
        }

        connectDeviceSequentially(0);
    }

    private void connectDeviceSequentially(int index) {
        if (index >= selectedDevices.size()) return;

        ImuDevice device = selectedDevices.get(index);
        String side = (index == 0) ? "left" : "right";

        MetaWearBoard board = btleBinder.getMetaWearBoard(device.getBluetoothDevice());
        device.setBoard(board);

        board.connectAsync().continueWith(task -> {
            if (!task.isFaulted()) {
                device.setConnected(true);
                runOnUiThread(() -> startStreamingForDevice(device, side));
                connectDeviceSequentially(index + 1);
            } else {
                Log.e("IMU", "Connection failed: " + device.getName() + " | ");
            }
            return null;
        });
    }

    private void startStreamingForDevice(ImuDevice device, String side) {
        String deviceName = device.getName() != null ? device.getName() : device.getMacAddress();

        if (streamService == null || device.getBoard() == null) return;

        streamService.startStreaming(
                device,
                side,
                gyroZ -> runOnUiThread(() -> {
                    if (side.equals("left")) {
                        device1Name.setText(deviceName);
                        device1GyroZ.setText(new StringBuilder().append("Gyro Z: ").append(gyroZ).toString());
                    } else {
                        device2Name.setText(deviceName);
                        device2GyroZ.setText(new StringBuilder().append("Gyro Z: ").append(gyroZ).toString());
                    }
                })
        );
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (selectedDevices != null) {
            for (ImuDevice device : selectedDevices) {
                if (device.getBoard() != null && device.getBoard().isConnected()) {
                    device.getBoard().disconnectAsync();
                }
            }
        }

        unbindService(btleConnection);
        unbindService(streamConnection);
    }
}
