package com.example.gaitrehabapp;

import android.Manifest;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Bundle;
import android.os.IBinder;
import android.widget.Button;
import android.widget.Toast;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresPermission;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.example.gaitrehabapp.adapters.DeviceAdapter;
import com.example.gaitrehabapp.models.ImuDevice;
import com.example.gaitrehabapp.services.ImuScannerService;

import java.util.ArrayList;
import java.util.List;

public class DeviceScanActivity extends AppCompatActivity implements ImuScannerService.ScanListener {

    private ImuScannerService scannerService;
    private DeviceAdapter adapter;

    private final ServiceConnection scannerConnection = new ServiceConnection() {
        @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            scannerService = ((ImuScannerService.LocalBinder) service).getService();
            scannerService.setScanListener(DeviceScanActivity.this);
            scannerService.startScan();
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
            scannerService = null;
        }
    };

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_device_scan);

        // Bind the scanner service
        Intent scannerIntent = new Intent(this, ImuScannerService.class);
        bindService(scannerIntent, scannerConnection, Context.BIND_AUTO_CREATE);

        // RecyclerView setup
        RecyclerView deviceList = findViewById(R.id.deviceList);
        deviceList.setLayoutManager(new LinearLayoutManager(this));
        adapter = new DeviceAdapter();
        deviceList.setAdapter(adapter);

        // Rescan Button
        Button rescanButton = findViewById(R.id.rescanButton);
        rescanButton.setOnClickListener(v -> {
            if (scannerService != null) {
                scannerService.startScan();
                Toast.makeText(this, "Scanning for devices...", Toast.LENGTH_SHORT).show();
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

    @Override
    public void onDeviceDiscovered(ImuDevice device) {
        runOnUiThread(() -> {
            List<ImuDevice> devices = scannerService.getDiscoveredDevices();
            adapter.updateDevices(devices);
        });
    }

    @Override
    public void onDeviceConnected(ImuDevice device) {
        // Optional feedback
    }

    @Override
    public void onConnectionFailed(ImuDevice device) {
        runOnUiThread(() ->
                Toast.makeText(this, "Failed to connect to " + device.getName(), Toast.LENGTH_SHORT).show()
        );
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        unbindService(scannerConnection);
    }
}
