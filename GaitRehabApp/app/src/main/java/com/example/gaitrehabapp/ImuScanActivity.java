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
import com.mbientlab.metawear.android.BtleService;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ImuScanActivity extends AppCompatActivity {
    private static final String TAG = "DeviceScanActivity";
    private boolean scanning = false;
    private BluetoothLeScanner bluetoothScanner;
    private BtleService.LocalBinder serviceBinder;
    private DeviceAdapter adapter;
    private final Map<String, ImuDevice> discoveredDevices = new HashMap<>();
    private final Set<String> seenAddresses = new HashSet<>();

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

    private final ScanCallback metaWearScanCallback = new ScanCallback() {
        @RequiresPermission(allOf = {Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT})
        @Override
        public void onScanResult(int callbackType, ScanResult result) {
            if (!scanning) return; // ignore late callbacks after stopScan()

            BluetoothDevice device = result.getDevice();
            if (device == null || device.getAddress() == null) return;
            if (seenAddresses.contains(device.getAddress())) return;

            String name = device.getName() != null ? device.getName() : "";
            boolean looksLikeMbient = name.contains("MetaWear") || name.contains("MetaMotion");
            if (!looksLikeMbient) return;

            seenAddresses.add(device.getAddress());

            ImuDevice imu = new ImuDevice(device);
            imu.setModel(name);
            runOnUiThread(() -> handleValidImuDevice(imu));
        }
    };

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_device_scan);

        Log.d(TAG, "DeviceScanActivity onCreate");

        // Bind BLE service (used later in ImuStreamActivity)
        bindService(new Intent(this, BtleService.class), btleConnection, Context.BIND_AUTO_CREATE);

        // RecyclerView
        RecyclerView deviceList = findViewById(R.id.deviceList);
        deviceList.setLayoutManager(new LinearLayoutManager(this));
        adapter = new DeviceAdapter();
        deviceList.setAdapter(adapter);

        // Rescan
        Button rescanButton = findViewById(R.id.rescanButton);
        rescanButton.setOnClickListener(v -> {
            if (bluetoothScanner != null) {
                stopScan();
                startScan();
                Toast.makeText(this, "Rescanning for devices...", Toast.LENGTH_SHORT).show();
            }
        });

        // Select
        Button selectButton = findViewById(R.id.selectButton);
        selectButton.setOnClickListener(v -> {
            List<ImuDevice> selectedDevices = adapter.getSelectedDevices();
            if (selectedDevices.isEmpty()) {
                Toast.makeText(this, "Select at least one device.", Toast.LENGTH_SHORT).show();
            } else if (selectedDevices.size() > 2) {
                Toast.makeText(this, "You can only select up to 2 devices.", Toast.LENGTH_SHORT).show();
            } else {
                stopScan(); // stop scanning before navigating
                Intent intent = new Intent(ImuScanActivity.this, ImuStreamActivity.class);
                intent.putParcelableArrayListExtra("selected_devices", new ArrayList<>(selectedDevices));
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
        if (bluetoothScanner == null || scanning) return;
        discoveredDevices.clear();
        seenAddresses.clear();
        scanning = true;
        bluetoothScanner.startScan(metaWearScanCallback);
        Log.d(TAG, "BLE scan started");
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    private void stopScan() {
        if (bluetoothScanner == null || !scanning) return;
        try {
            bluetoothScanner.stopScan(metaWearScanCallback);
        } catch (Exception ignored) {}
        scanning = false;
        Log.d(TAG, "BLE scan stopped");
    }

    private void handleValidImuDevice(ImuDevice imu) {
        String address = imu.getMacAddress();
        if (!discoveredDevices.containsKey(address)) {
            discoveredDevices.put(address, imu);
            adapter.updateDevices(new ArrayList<>(discoveredDevices.values()));
            // keep toasts minimal during scans to avoid noise
            // Toast.makeText(this, "Discovered: " + imu.getModel(), Toast.LENGTH_SHORT).show();
        }
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    @Override
    protected void onPause() {
        super.onPause();
        stopScan();
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopScan();
        try { unbindService(btleConnection); } catch (Exception ignored) {}
    }
}