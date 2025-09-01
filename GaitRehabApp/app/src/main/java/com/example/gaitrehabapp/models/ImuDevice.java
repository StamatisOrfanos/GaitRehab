package com.example.gaitrehabapp.models;

import android.Manifest;
import android.bluetooth.BluetoothDevice;
import android.os.Parcel;
import android.os.Parcelable;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresPermission;
import com.mbientlab.metawear.MetaWearBoard;

public class ImuDevice implements Parcelable {
    private final String name;
    private String model;
    private final String macAddress;
    private final BluetoothDevice bluetoothDevice;
    private boolean isConnected;
    private String role;
    private DeviceType deviceType;
    private transient MetaWearBoard board;

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    public ImuDevice(BluetoothDevice device) {
        this.name = device.getName();
        this.macAddress = device.getAddress();
        this.bluetoothDevice = device;
        this.isConnected = false;
        this.role = null;
        this.deviceType = detectTypeFromName(name);
    }

    private static DeviceType detectTypeFromName(@Nullable String name) {
        if (name == null) return DeviceType.UNKNOWN;
        String n = name.toUpperCase();
        if (n.contains("METAWEAR") || n.contains("METAMOTION")) return DeviceType.METAWEAR;
        if (n.startsWith("WT901") || n.contains("WITMOTION") || n.contains("BWT901BLE")) return DeviceType.WITMOTION;
        return DeviceType.UNKNOWN;
    }

    protected ImuDevice(Parcel in) {
        name = in.readString();
        macAddress = in.readString();
        bluetoothDevice = in.readParcelable(BluetoothDevice.class.getClassLoader());
        isConnected = in.readByte() != 0;
        role = in.readString();
        model = in.readString();
        String typeName = in.readString();
        try {
            deviceType = typeName != null ? DeviceType.valueOf(typeName) : DeviceType.UNKNOWN;
        } catch (IllegalArgumentException e) {
            deviceType = DeviceType.UNKNOWN;
        }
    }

    public static final Creator<ImuDevice> CREATOR = new Creator<ImuDevice>() {
        @Override public ImuDevice createFromParcel(Parcel in) { return new ImuDevice(in); }
        @Override public ImuDevice[] newArray(int size) { return new ImuDevice[size]; }
    };

    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeString(name);
        dest.writeString(macAddress);
        dest.writeParcelable(bluetoothDevice, flags);
        dest.writeByte((byte) (isConnected ? 1 : 0));
        dest.writeString(role);
        dest.writeString(model);
        dest.writeString(deviceType.name());
    }



    public String getName() {
        return name;
    }
    public String getModel() {
        return model;
    }
    public void setModel(String model) {
        this.model = model;
    }
    public String getMacAddress() {
        return macAddress;
    }
    public BluetoothDevice getBluetoothDevice() {
        return bluetoothDevice;
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
    public MetaWearBoard getBoard() {
        return board;
    }
    public void setBoard(MetaWearBoard board) {
        this.board = board;
    }
    public DeviceType getDeviceType() { return deviceType; }
    public void setDeviceType(DeviceType type) {
        this.deviceType = (type != null) ? type : DeviceType.UNKNOWN;
    }
    public boolean isMetaWear() { return deviceType == DeviceType.METAWEAR; }
    public boolean isWitMotion() { return deviceType == DeviceType.WITMOTION; }
}
