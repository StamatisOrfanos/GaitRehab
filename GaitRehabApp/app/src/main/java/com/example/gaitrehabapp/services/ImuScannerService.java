package com.example.gaitrehabapp.services;

import android.Manifest;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanResult;
import android.content.Context;
import android.content.Intent;
import android.os.Binder;
import android.os.IBinder;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresPermission;

import com.example.gaitrehabapp.models.ImuDevice;
import com.mbientlab.metawear.MetaWearBoard;
import com.mbientlab.metawear.android.BtleService;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ImuScannerService extends android.app.Service {
    private final IBinder binder = new LocalBinder();

    public interface ScanListener {
        void onDeviceDiscovered(ImuDevice device);
        void onDeviceConnected(ImuDevice device);
        void onConnectionFailed(ImuDevice device);
    }

    private ScanListener scanListener;
    private BluetoothLeScanner scanner;
    private BluetoothAdapter adapter;
    private BtleService.LocalBinder btleBinder;

    private final Map<String, ImuDevice> discoveredDevices = new HashMap<>();

    public class LocalBinder extends Binder {
        public ImuScannerService getService() {
            return ImuScannerService.this;
        }
    }

    public void setScanListener(ScanListener listener) {
        this.scanListener = listener;
    }

    public void initialize(Context context, BtleService.LocalBinder btleBinder) {
        this.btleBinder = btleBinder;
        BluetoothManager manager = (BluetoothManager) context.getSystemService(Context.BLUETOOTH_SERVICE);
        adapter = manager.getAdapter();
        scanner = adapter.getBluetoothLeScanner();
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    public void startScan() {
        discoveredDevices.clear();
        scanner.startScan(scanCallback);
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    public void stopScan() {
        scanner.stopScan(scanCallback);
    }

    public List<ImuDevice> getDiscoveredDevices() {
        return new ArrayList<>(discoveredDevices.values());
    }

    public void connectToDevice(ImuDevice imuDevice) {
        MetaWearBoard board = btleBinder.getMetaWearBoard(imuDevice.getBluetoothDevice());
        imuDevice.setBoard(board);

        board.connectAsync().continueWith(task -> {
            imuDevice.setConnected(!task.isFaulted());
            if (scanListener != null) {
                if (task.isFaulted()) {
                    scanListener.onConnectionFailed(imuDevice);
                } else {
                    scanListener.onDeviceConnected(imuDevice);
                }
            }
            return null;
        });
    }

    private final ScanCallback scanCallback = new ScanCallback() {
        @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
        @Override
        public void onScanResult(int callbackType, ScanResult result) {
            BluetoothDevice device = result.getDevice();
            if (device.getName() != null && device.getName().contains("MetaWear")) {
                String mac = device.getAddress();
                if (!discoveredDevices.containsKey(mac)) {
                    ImuDevice imu = new ImuDevice(device);
                    discoveredDevices.put(mac, imu);
                    if (scanListener != null) {
                        scanListener.onDeviceDiscovered(imu);
                    }
                }
            }
        }
    };

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }
}
