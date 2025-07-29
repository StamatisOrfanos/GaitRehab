package com.example.gaitrehabapp.Models;

import android.Manifest;
import android.bluetooth.BluetoothDevice;

import androidx.annotation.RequiresPermission;

import com.mbientlab.metawear.MetaWearBoard;

public class ImuDevice {
    private final String name;
    private final String macAddress;
    private final BluetoothDevice bluetoothDevice;
    private MetaWearBoard board;
    private boolean isConnected;
    private String role;

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    public ImuDevice(BluetoothDevice device) {
        this.name = device.getName();
        this.macAddress = device.getAddress();
        this.bluetoothDevice = device;
        this.isConnected = false;
        this.role = null;
    }

    public String getName() {
        return name;
    }

    public String getMacAddress() {
        return macAddress;
    }

    public BluetoothDevice getBluetoothDevice() {
        return bluetoothDevice;
    }

    public MetaWearBoard getBoard() {
        return board;
    }

    public void setBoard(MetaWearBoard board) {
        this.board = board;
    }

    public boolean isConnected() {
        return isConnected;
    }

    public void setConnected(boolean connected) {
        isConnected = connected;
    }

    public String getRole() {
        return role;
    }

    public void setRole(String role) {
        this.role = role;
    }
}
