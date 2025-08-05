package com.example.gaitrehabapp.models;

public class DataPoint {
    public final long timestamp;
    public final float z;

    public DataPoint(float z, long timestamp) {
        this.z = z;
        this.timestamp = timestamp;
    }
}

