package com.example.gaitrehabapp.models;

import java.util.LinkedList;

public class CircularBuffer {
    private final int capacity;
    private final LinkedList<Float> zValues;
    private final LinkedList<Long> timestamps;

    public CircularBuffer(int capacity) {
        this.capacity = capacity;
        this.zValues = new LinkedList<>();
        this.timestamps = new LinkedList<>();
    }

    public void add(float z, long timestamp) {
        if (zValues.size() >= capacity) {
            zValues.removeFirst();
            timestamps.removeFirst();
        }
        zValues.add(z);
        timestamps.add(timestamp);
    }

    public float[] getZArray() {
        float[] array = new float[zValues.size()];
        for (int i = 0; i < zValues.size(); i++) {
            array[i] = zValues.get(i);
        }
        return array;
    }

    public long[] getTimestampArray() {
        long[] array = new long[timestamps.size()];
        for (int i = 0; i < timestamps.size(); i++) {
            array[i] = timestamps.get(i);
        }
        return array;
    }

    public int size() {
        return zValues.size();
    }
}