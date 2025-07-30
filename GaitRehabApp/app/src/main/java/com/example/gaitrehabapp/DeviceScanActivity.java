package com.example.gaitrehabapp;

import android.Manifest;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.location.LocationManager;
import android.os.Build;
import android.os.Bundle;
import android.os.IBinder;
import android.provider.Settings;
import android.util.Log;
import android.widget.Button;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresPermission;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.example.gaitrehabapp.models.ImuDevice;
import com.example.gaitrehabapp.services.ImuScannerService;
import com.mbientlab.metawear.android.BtleService;
import java.util.ArrayList;
import java.util.List;

public class DeviceScanActivity extends AppCompatActivity {

    private static final int PERMISSION_REQUEST_CODE = 1;
    private static final String TAG = "DeviceScanActivity";
    private BtleService.LocalBinder btleBinder;
    private ImuScannerService scannerService;
    private final List<ImuDevice> discovered = new ArrayList<>();
    private DeviceAdapter adapter;

    private final ServiceConnection btleConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            Log.d(TAG, "BtleService connected");
            btleBinder = (BtleService.LocalBinder) service;
            bindService(new Intent(DeviceScanActivity.this, ImuScannerService.class),
                    scannerConnection, Context.BIND_AUTO_CREATE);
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            Log.d(TAG, "BtleService disconnected");
            btleBinder = null;
        }
    };

    private final ServiceConnection scannerConnection = new ServiceConnection() {
        @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            Log.d(TAG, "ImuScannerService connected");

            scannerService = ((ImuScannerService.LocalBinder) service).getService();
            scannerService.initialize(DeviceScanActivity.this, btleBinder);
            scannerService.setScanListener(new ImuScannerService.ScanListener() {
                @Override
                public void onDeviceDiscovered(ImuDevice device) {
                    runOnUiThread(() -> {
                        Log.i(TAG, "Discovered device: " + device.getName() + " [" + device.getMacAddress() + "]");
                        discovered.add(device);
                        adapter.notifyItemInserted(discovered.size() - 1);
                    });
                }

                @Override
                public void onDeviceConnected(ImuDevice device) {
                    runOnUiThread(() -> {
                        Toast.makeText(DeviceScanActivity.this, "Connected to: " + device.getMacAddress(), Toast.LENGTH_SHORT).show();
                        Log.d("DeviceScanActivity", "Connected to: " + device.getMacAddress());
                        });
                }

                @Override
                public void onConnectionFailed(ImuDevice device) {
                    runOnUiThread(() -> {
                        Toast.makeText(DeviceScanActivity.this, "Failed to connect: " + device.getMacAddress(), Toast.LENGTH_SHORT).show();
                            Log.e("DeviceScanActivity", "Failed to connect: " + device.getMacAddress());
                        });
                    }
                });

            if (ActivityCompat.checkSelfPermission(DeviceScanActivity.this, Manifest.permission.BLUETOOTH_SCAN)
                    == PackageManager.PERMISSION_GRANTED) {
                scannerService.startScan();
                Log.d(TAG, "Started BLE scan");
            } else {
                Toast.makeText(DeviceScanActivity.this, "Missing BLUETOOTH_SCAN permission", Toast.LENGTH_LONG).show();
            }
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            Log.d(TAG, "ImuScannerService disconnected");
            scannerService = null;
        }
    };

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_device_scan);
        Log.d(TAG, "onCreate called");

        if (!isLocationEnabled()) {
            Toast.makeText(this, "Please enable Location Services for BLE scanning", Toast.LENGTH_LONG).show();
            startActivity(new Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS));
            return;
        }

        setupRecyclerView();
        setupSelectButton();
        requestPermissions();
    }

    private void setupRecyclerView() {
        RecyclerView recycler = findViewById(R.id.deviceList);
        recycler.setLayoutManager(new LinearLayoutManager(this));
        adapter = new DeviceAdapter(discovered);
        recycler.setAdapter(adapter);
        Log.d(TAG, "RecyclerView initialized");
    }

    private void setupSelectButton() {
        Button selectBtn = findViewById(R.id.selectButton);
        selectBtn.setOnClickListener(v -> {
            List<ImuDevice> selected = adapter.getSelectedDevices();
            Log.d(TAG, "Selected " + selected.size() + " devices");

            if (selected.isEmpty() || selected.size() > 2) {
                Toast.makeText(this, "Please select 1 or 2 devices", Toast.LENGTH_SHORT).show();
                return;
            }

            Toast.makeText(this, "Selected " + selected.size() + " device(s)", Toast.LENGTH_SHORT).show();
            // TODO: pass selected devices to next activity or stream service
        });
    }

    private void requestPermissions() {
        if (hasPermissions()) {
            Log.d(TAG, "Permissions already granted. Binding BtleService...");
            bindService(new Intent(this, BtleService.class), btleConnection, Context.BIND_AUTO_CREATE);
        } else {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                ActivityCompat.requestPermissions(this,
                        new String[]{
                                Manifest.permission.BLUETOOTH_SCAN,
                                Manifest.permission.BLUETOOTH_CONNECT,
                                Manifest.permission.ACCESS_FINE_LOCATION
                        },
                        PERMISSION_REQUEST_CODE);
            } else {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.ACCESS_FINE_LOCATION},
                        PERMISSION_REQUEST_CODE);
            }
        }
    }

    private boolean hasPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            return ActivityCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED &&
                    ActivityCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED &&
                    ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED;
        } else {
            return ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED;
        }
    }

    private boolean isLocationEnabled() {
        LocationManager locationManager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);
        return locationManager != null && locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (hasPermissions()) {
            Log.d(TAG, "Permissions granted after request");
            bindService(new Intent(this, BtleService.class), btleConnection, Context.BIND_AUTO_CREATE);
        } else {
            Log.w(TAG, "Permissions denied");
            Toast.makeText(this, "Permissions are required to scan for IMUs.", Toast.LENGTH_LONG).show();
            finish();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (scannerService != null) {
            unbindService(scannerConnection);
            Log.d(TAG, "Unbound scanner service");
        }
        if (btleBinder != null) {
            unbindService(btleConnection);
            Log.d(TAG, "Unbound BTLE service");
        }
    }
}
