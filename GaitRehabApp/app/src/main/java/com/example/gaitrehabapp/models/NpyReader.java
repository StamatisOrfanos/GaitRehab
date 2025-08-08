package com.example.gaitrehabapp.models;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;

public class NpyReader {
    public float[] readFloatArray(InputStream is) throws IOException {
        DataInputStream dis = new DataInputStream(is);
        byte[] header = new byte[128]; dis.readFully(header);
        int len = (int) (is.available() / 4.0);
        float[] data = new float[len];
        for (int i = 0; i < len; i++) {
            data[i] = Float.intBitsToFloat(Integer.reverseBytes(dis.readInt()));
        }
        return data;
    }
}
