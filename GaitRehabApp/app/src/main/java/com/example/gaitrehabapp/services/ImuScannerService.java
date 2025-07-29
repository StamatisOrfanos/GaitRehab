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
import com.mbientlab.metawear.MetaWearBoard;
import com.mbientlab.metawear.android.BtleService;

public class ImuScannerService extends android.app.Service {
    private final IBinder binder = new LocalBinder();

    public interface ConnectionListener {
        void onDeviceFound(BluetoothDevice device);
        void onConnected(BluetoothDevice device);
        void onConnectionFailed(BluetoothDevice device);
    }

    private ConnectionListener connectionListener;
    private BluetoothLeScanner scanner;
    private BluetoothAdapter adapter;
    private BtleService.LocalBinder btleBinder;
    private MetaWearBoard metaWearBoard;

    public class LocalBinder extends Binder {
        public ImuScannerService getService() {
            return ImuScannerService.this;
        }
    }

    public void setConnectionListener(ConnectionListener listener) {
        this.connectionListener = listener;
    }

    public void initialize(Context context, BtleService.LocalBinder btleBinder) {
        this.btleBinder = btleBinder;
        BluetoothManager manager = (BluetoothManager) context.getSystemService(Context.BLUETOOTH_SERVICE);
        adapter = manager.getAdapter();
        scanner = adapter.getBluetoothLeScanner();
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    public void scanForDevices() {
        scanner.startScan(scanCallback);
    }

    private final ScanCallback scanCallback = new ScanCallback() {
        @RequiresPermission(allOf = {Manifest.permission.BLUETOOTH_CONNECT, Manifest.permission.BLUETOOTH_SCAN})
        @Override
        public void onScanResult(int callbackType, ScanResult result) {
            BluetoothDevice device = result.getDevice();
            if (device.getName() != null && device.getName().contains("MetaWear")) {
                scanner.stopScan(this);
                if (connectionListener != null) {
                    connectionListener.onDeviceFound(device);
                }
                connectToDevice(device);
            }
        }
    };

    private void connectToDevice(BluetoothDevice device) {
        metaWearBoard = btleBinder.getMetaWearBoard(device);
        metaWearBoard.connectAsync().continueWith(task -> {
            if (connectionListener != null) {
                if (task.isFaulted()) {
                    connectionListener.onConnectionFailed(device);
                } else {
                    connectionListener.onConnected(device);
                }
            }
            return null;
        });
    }

    public MetaWearBoard getBoard() {
        return metaWearBoard;
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }
}
