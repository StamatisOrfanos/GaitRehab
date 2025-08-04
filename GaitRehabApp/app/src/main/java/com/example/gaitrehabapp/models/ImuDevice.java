package com.example.gaitrehabapp.models;

import android.Manifest;
import android.bluetooth.BluetoothDevice;
import android.os.Parcel;
import android.os.Parcelable;
import androidx.annotation.RequiresPermission;
import com.mbientlab.metawear.MetaWearBoard;

public class ImuDevice implements Parcelable {
    private final String name;
    private String model;
    private final String macAddress;
    private final BluetoothDevice bluetoothDevice;
    private boolean isConnected;
    private String role;
    private transient MetaWearBoard board;

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    public ImuDevice(BluetoothDevice device) {
        this.name = device.getName();
        this.macAddress = device.getAddress();
        this.bluetoothDevice = device;
        this.isConnected = false;
        this.role = null;
    }

    protected ImuDevice(Parcel in) {
        name = in.readString();
        macAddress = in.readString();
        bluetoothDevice = in.readParcelable(BluetoothDevice.class.getClassLoader());
        isConnected = in.readByte() != 0;
        role = in.readString();
        model = in.readString();
    }

    public static final Creator<ImuDevice> CREATOR = new Creator<ImuDevice>() {
        @Override
        public ImuDevice createFromParcel(Parcel in) {
            return new ImuDevice(in);
        }

        @Override
        public ImuDevice[] newArray(int size) {
            return new ImuDevice[size];
        }
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
    }

    public String getName() { return name; }

    public String getModel() { return model; }

    public void setModel(String model) { this.model = model; }

    public String getMacAddress() {
        return macAddress;
    }

    public BluetoothDevice getBluetoothDevice() {
        return bluetoothDevice;
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

    public MetaWearBoard getBoard() {
        return board;
    }

    public void setBoard(MetaWearBoard board) {
        this.board = board;
    }
}
