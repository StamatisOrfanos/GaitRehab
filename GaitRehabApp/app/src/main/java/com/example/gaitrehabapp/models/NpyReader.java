package com.example.gaitrehabapp.models;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class NpyReader {
    public float[] readFloatArray(InputStream is) throws IOException {
        DataInputStream dis = new DataInputStream(new BufferedInputStream(is));
        byte[] magic = new byte[6];
        dis.readFully(magic);
        if (!(magic[0]==(byte)0x93 && magic[1]=='N' && magic[2]=='U' && magic[3]=='M' && magic[4]=='P' && magic[5]=='Y')) {
            throw new IOException("Not an .npy file");
        }

        int major = dis.readUnsignedByte();

        int headerLen;
        if (major == 1) {
            headerLen = Short.toUnsignedInt(Short.reverseBytes(dis.readShort()));
        } else if (major == 2) {
            headerLen = Integer.reverseBytes(dis.readInt());
        } else {
            headerLen = Integer.reverseBytes(dis.readInt()); // v3
        }

        byte[] headerBytes = new byte[headerLen];
        dis.readFully(headerBytes);
        String header = new String(headerBytes, "ASCII").trim();

        String descr = extract(header, "'descr':\\s*'([^']+)'");
        String shapeStr = extract(header, "'shape':\\s*\\(([^\\)]*)\\)");
        if (descr == null || shapeStr == null) throw new IOException("Invalid .npy header");

        boolean littleEndian = descr.charAt(0) == '<' || descr.charAt(0) == '|';
        int bytesPer = 0;
        if (descr.endsWith("f8")) {
            bytesPer = 8; }
        else if (descr.endsWith("f4")) {
            bytesPer = 4; }
        else throw new IOException("Unsupported dtype: " + descr);

        int count = 1;
        if (!shapeStr.trim().isEmpty()) {
            for (String s : shapeStr.split(",")) {
                s = s.trim();
                if (s.isEmpty()) continue;
                count *= Integer.parseInt(s);
            }
        }

        byte[] data = new byte[count * bytesPer];
        dis.readFully(data);

        ByteBuffer bb = ByteBuffer.wrap(data);
        bb.order(littleEndian ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN);

        float[] out = new float[count];
        if (bytesPer == 8) {
            for (int i = 0; i < count; i++) out[i] = (float) bb.getDouble();
        } else {
            for (int i = 0; i < count; i++) out[i] = bb.getFloat();
        }
        return out;
    }

    private static String extract(String s, String regex) {
        Matcher m = Pattern.compile(regex).matcher(s);
        return m.find() ? m.group(1) : null;
    }
}
