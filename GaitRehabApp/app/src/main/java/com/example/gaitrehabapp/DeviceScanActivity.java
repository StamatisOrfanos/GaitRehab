package com.example.gaitrehabapp;

import android.Manifest;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanResult;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Bundle;
import android.os.IBinder;
import android.util.Log;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresPermission;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.gaitrehabapp.adapters.DeviceAdapter;
import com.example.gaitrehabapp.models.ImuDevice;
import com.mbientlab.metawear.MetaWearBoard;
import com.mbientlab.metawear.android.BtleService;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DeviceScanActivity extends AppCompatActivity {

    private static final String TAG = "DeviceScanActivity";
    private BtleService.LocalBinder serviceBinder;

    private BluetoothLeScanner bluetoothScanner;
    private DeviceAdapter adapter;
    private final Map<String, ImuDevice> discoveredDevices = new HashMap<>();

    private final ServiceConnection btleConnection = new ServiceConnection() {
        @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            Log.d(TAG, "BtleService connected");
            serviceBinder = (BtleService.LocalBinder) service;
            setupScanner();
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            serviceBinder = null;
        }
    };

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_device_scan);

        Log.d(TAG, "DeviceScanActivity onCreate");

        // Bind the BtleService
        bindService(new Intent(this, BtleService.class), btleConnection, Context.BIND_AUTO_CREATE);

        // RecyclerView setup
        RecyclerView deviceList = findViewById(R.id.deviceList);
        deviceList.setLayoutManager(new LinearLayoutManager(this));
        adapter = new DeviceAdapter();
        deviceList.setAdapter(adapter);

        // Rescan Button
        Button rescanButton = findViewById(R.id.rescanButton);
        rescanButton.setOnClickListener(v -> {
            if (bluetoothScanner != null) {
                startScan();
                Toast.makeText(this, "Rescanning for devices...", Toast.LENGTH_SHORT).show();
            }
        });

        // Select Devices Button
        Button selectButton = findViewById(R.id.selectButton);
        selectButton.setOnClickListener(v -> {
            List<ImuDevice> selected = adapter.getSelectedDevices();
            if (selected.isEmpty()) {
                Toast.makeText(this, "Select at least one device.", Toast.LENGTH_SHORT).show();
            } else if (selected.size() > 2) {
                Toast.makeText(this, "You can only select up to 2 devices.", Toast.LENGTH_SHORT).show();
            } else {
                Intent intent = new Intent(this, ImuStreamActivity.class);
                intent.putParcelableArrayListExtra("selected_devices", new ArrayList<>(selected));
                startActivity(intent);
            }
        });
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    private void setupScanner() {
        BluetoothManager bluetoothManager = (BluetoothManager) getSystemService(Context.BLUETOOTH_SERVICE);
        BluetoothAdapter bluetoothAdapter = bluetoothManager.getAdapter();
        bluetoothScanner = bluetoothAdapter.getBluetoothLeScanner();
        startScan();
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    private void startScan() {
        discoveredDevices.clear();
        bluetoothScanner.startScan(scanCallback);
    }

    private final ScanCallback scanCallback = new ScanCallback() {
        @RequiresPermission(allOf = {Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT})
        @Override
        public void onScanResult(int callbackType, ScanResult result) {
            BluetoothDevice device = result.getDevice();
            String name = device.getName();
            String address = device.getAddress();

            if (name != null && name.contains("MetaWear") && !discoveredDevices.containsKey(address)) {
                bluetoothScanner.stopScan(this);

                MetaWearBoard board = serviceBinder.getMetaWearBoard(device);
                board.connectAsync().continueWith(task -> {
                    if (!task.isFaulted()) {
                        ImuDevice imuDevice = new ImuDevice(device);
                        discoveredDevices.put(address, imuDevice);

                        runOnUiThread(() -> {
                            adapter.updateDevices(new ArrayList<>(discoveredDevices.values()));
                            Toast.makeText(DeviceScanActivity.this, "Connected to " + name, Toast.LENGTH_SHORT).show();
                        });
                    } else {
                        runOnUiThread(() ->
                                Toast.makeText(DeviceScanActivity.this, "Failed to connect to " + name, Toast.LENGTH_SHORT).show()
                        );
                    }
                    return null;
                });
            }
        }
    };

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    @Override
    protected void onDestroy() {
        super.onDestroy();
        bluetoothScanner.stopScan(scanCallback);
        unbindService(btleConnection);
    }
}