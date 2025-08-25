// com/example/gaitrehabapp/models/CircularBuffer.java
package com.example.gaitrehabapp.models;

public class CircularBuffer {
    private final int capacity;
    private final float[] zRing;
    private final long[] tRing;
    private int head = 0;        // next write index
    private int size = 0;        // number of valid samples (<= capacity)

    public CircularBuffer(int capacity) {
        this.capacity = Math.max(1, capacity);
        this.zRing = new float[this.capacity];
        this.tRing = new long[this.capacity];
    }

    public synchronized void add(float z, long ts) {
        zRing[head] = z;
        tRing[head] = ts;
        head = (head + 1) % capacity;
        if (size < capacity) size++;
    }

    public synchronized int size() { return size; }

    public synchronized Snapshot snapshot() {
        float[] zOut = new float[size];
        long[]  tOut = new long[size];
        int start = (head - size + capacity) % capacity;
        for (int i = 0; i < size; i++) {
            int idx = (start + i) % capacity;
            zOut[i] = zRing[idx];
            tOut[i] = tRing[idx];
        }
        return new Snapshot(zOut, tOut);
    }

    public synchronized float[] getZArray() {
        return snapshot().z;
    }
    public synchronized long[] getTimestampArray() {
        return snapshot().t;
    }

    public record Snapshot(float[] z, long[] t) {
    }
}
