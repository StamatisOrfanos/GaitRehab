package com.example.gaitrehabapp;

import android.Manifest;
import android.bluetooth.BluetoothDevice;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Build;
import android.os.Bundle;
import android.os.IBinder;
import android.widget.Toast;

import androidx.annotation.RequiresPermission;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.example.gaitrehabapp.services.ImuScannerService;
import com.mbientlab.metawear.android.BtleService;

public class MainActivity extends AppCompatActivity {
    private ImuScannerService imuScannerService;
    private BtleService.LocalBinder btleBinder;

    private final ServiceConnection imuServiceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            imuScannerService = ((ImuScannerService.LocalBinder) service).getService();
            imuScannerService.setConnectionListener(connectionListener);
            imuScannerService.initialize(MainActivity.this, btleBinder);
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            imuScannerService = null;
        }
    };

    private final ServiceConnection btleServiceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            btleBinder = (BtleService.LocalBinder) service;

            // Bind IMU scanner after BTLE is ready
            Intent imuIntent = new Intent(MainActivity.this, ImuScannerService.class);
            bindService(imuIntent, imuServiceConnection, Context.BIND_AUTO_CREATE);
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            btleBinder = null;
        }
    };

    private final ImuScannerService.ConnectionListener connectionListener = new ImuScannerService.ConnectionListener() {
        @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
        @Override
        public void onDeviceFound(BluetoothDevice device) {
            Toast.makeText(MainActivity.this, "Found: " + device.getName(), Toast.LENGTH_SHORT).show();
        }

        @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
        @Override
        public void onConnected(BluetoothDevice device) {
            Toast.makeText(MainActivity.this, "Connected to " + device.getName(), Toast.LENGTH_SHORT).show();
        }

        @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
        @Override
        public void onConnectionFailed(BluetoothDevice device) {
            Toast.makeText(MainActivity.this, "Connection failed: " + device.getName(), Toast.LENGTH_SHORT).show();
        }
    };

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        requestPermissions();

        bindService(new Intent(this, BtleService.class), btleServiceConnection, Context.BIND_AUTO_CREATE);

        findViewById(R.id.scanButton).setOnClickListener(v -> {
            if (imuScannerService != null) {
                imuScannerService.scanForDevices();
            }
        });
    }

    private void requestPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ActivityCompat.requestPermissions(this,
                    new String[]{
                            Manifest.permission.BLUETOOTH_SCAN,
                            Manifest.permission.BLUETOOTH_CONNECT,
                            Manifest.permission.ACCESS_FINE_LOCATION
                    },
                    1);
        } else {
            ActivityCompat.requestPermissions(this,
                    new String[]{
                            Manifest.permission.ACCESS_FINE_LOCATION
                    },
                    1);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        unbindService(imuServiceConnection);
        unbindService(btleServiceConnection);
    }
}
