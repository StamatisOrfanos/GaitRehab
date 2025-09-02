package com.example.gaitrehabapp.services;

import static com.example.gaitrehabapp.services.GaitFeatureExtractorService.featureExtraction;
import android.Manifest;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothProfile;
import android.content.Intent;
import android.media.AudioAttributes;
import android.media.Ringtone;
import android.media.RingtoneManager;
import android.net.Uri;
import android.os.Binder;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.os.IBinder;
import android.os.SystemClock;
import android.util.Log;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.annotation.RequiresPermission;
import androidx.core.app.NotificationCompat;
import com.example.gaitrehabapp.models.CircularBuffer;
import com.example.gaitrehabapp.models.DataPoint;
import com.example.gaitrehabapp.models.DeviceType;
import com.example.gaitrehabapp.models.GaitWindowResult;
import com.example.gaitrehabapp.models.ImuDevice;
import com.example.gaitrehabapp.models.ModelPredictor;
import com.mbientlab.metawear.data.AngularVelocity;
import com.mbientlab.metawear.module.Gyro;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public class ImuStreamService extends Service {
    private static final String TAG = "IMU_STREAM_SERVICE";

    private static final int FS_HZ = 100;
    private static final int WINDOW_MS = 2000;
    private static final int HOP_MS = 1000;
    private static final int WINDOW_SAMPLES = (FS_HZ * WINDOW_MS) / 1000;
    private static final int ANALYSIS_INTERVAL_MS = HOP_MS;
    private static final int BUFFER_CAPACITY = WINDOW_SAMPLES * 3;
    private static final long INFERENCE_COOLDOWN_MS = HOP_MS;
    private static final long ALERT_DURATION_MS = 300L;
    private static final long ALERT_COOLDOWN_MS = 1000L;
    private static final String ALERT_CHANNEL_ID = "gait_alert_channel";
    private static final UUID WIT_SERVICE_UUID_9B = UUID.fromString("0000ffe5-0000-1000-8000-00805f9b34fb");
    private static final UUID WIT_SERVICE_UUID_9A = UUID.fromString("0000ffe5-0000-1000-8000-00805f9a34fb");
    private static final UUID WIT_NOTIFY_UUID_9B  = UUID.fromString("0000ffe4-0000-1000-8000-00805f9b34fb");
    private static final UUID WIT_NOTIFY_UUID_9A  = UUID.fromString("0000ffe4-0000-1000-8000-00805f9a34fb");
    private static final UUID CCCD_UUID_2902_9B   = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb");
    private static final UUID CCCD_UUID_2902_9A   = UUID.fromString("00002902-0000-1000-8000-00805f9a34fb");
    private static final UUID CCCD_UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb");

    private static final boolean SEND_WIT_START_CMD = false;
    private static final byte[] WIT_START_CMD = new byte[]{(byte) 0xFF, (byte) 0xAA, 0x00, 0x00, 0x00};
    private static final int GATT_OP_WRITE_CCCD = 1;
    public class LocalBinder extends Binder { public ImuStreamService getService() { return ImuStreamService.this; } }
    @Nullable @Override public IBinder onBind(Intent intent) { return binder; }
    private final IBinder binder = new LocalBinder();
    private final Handler analysisHandler = new Handler();
    private final Map<String, CircularBuffer> bufferMap = new HashMap<>();
    private final Map<String, Boolean> pausedMap = new HashMap<>();
    private final Map<String, String> deviceToSideMap = new HashMap<>();
    private final Map<String, StringBuilder> sessionBuffers = new HashMap<>();
    private boolean analysisStarted = false;
    private long lastInferenceTs = 0L;
    private long lastAlertTs = 0L;
    private long lastProcessedEndTs = 0L;
    private ModelPredictor predictor;
    private NotificationManager notificationManager;
    private Ringtone alarmRingtone;
    private final Map<String, BluetoothGatt> gattMap = new HashMap<>();
    private final Map<String, Deque<Byte>> frameQueues = new HashMap<>();
    private final Map<String, Integer> pendingGattOp = new HashMap<>();
    private static final byte[][] WIT_START_CMDS = new byte[][]{
            {(byte)0xFF, (byte)0xAA, 0x00, 0x00, 0x00},
            {(byte)0xFF, (byte)0xAA, 0x52, 0x02, 0x00},
            {(byte)0xFF, (byte)0xAA, 0x69, (byte)0x88, (byte)0xB5}
    };

    @RequiresApi(api = Build.VERSION_CODES.TIRAMISU)
    @Override public void onCreate() {
        super.onCreate();
        try {
            Log.d(TAG, "Initializing ModelPredictor…");
            predictor = new ModelPredictor(getApplicationContext());
            Log.d(TAG, "ModelPredictor initialized successfully");
        } catch (Throwable t) {
            Log.e(TAG, "Failed to initialize ModelPredictor", t);
        }

        notificationManager = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        NotificationChannel channel = new NotificationChannel(ALERT_CHANNEL_ID, "Gait Alerts", NotificationManager.IMPORTANCE_HIGH);
        channel.setDescription("Alerts when abnormal gait is detected");
        channel.setSound(null, null);
        notificationManager.createNotificationChannel(channel);
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    @Override public void onDestroy() {
        super.onDestroy();
        analysisHandler.removeCallbacksAndMessages(null);
        stopAlarmTone();
        for (BluetoothGatt g : gattMap.values()) {
            try { g.disconnect(); g.close(); } catch (Exception ignored) {}
        }
        gattMap.clear();
        analysisStarted = false;
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    public void startStreaming(ImuDevice device, String side, GyroZCallback zCallback) {
        if (device == null) { Log.e(TAG, "startStreaming: device is null"); return; }
        String id = device.getMacAddress();
        deviceToSideMap.put(id, side == null ? "" : side.toLowerCase());
        pausedMap.put(id, false);

        if (device.getDeviceType() == DeviceType.METAWEAR) {
            startMetaWearStreaming(device, zCallback);
        } else if (device.getDeviceType() == DeviceType.WITMOTION) {
            startWitMotionStreaming(device, zCallback);
        } else {
            Log.w(TAG, "Unknown device type; cannot stream: " + device.getModel());
        }
    }

    public void pauseStreaming(ImuDevice device) { if (device != null) pausedMap.put(device.getMacAddress(), true); }
    public void resumeStreaming(ImuDevice device) { if (device != null) pausedMap.put(device.getMacAddress(), false); }

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    public void stopStreaming(ImuDevice device) {
        if (device == null) return;
        String id = device.getMacAddress();
        try {
            if (device.getDeviceType() == DeviceType.METAWEAR && device.getBoard() != null) {
                device.getBoard().getModule(Gyro.class).angularVelocity().stop();
                device.getBoard().getModule(Gyro.class).stop();
            } else if (device.getDeviceType() == DeviceType.WITMOTION) {
                BluetoothGatt g = gattMap.remove(id);
                if (g != null) { g.disconnect(); g.close(); }
            }
            Log.i(TAG, "Stopped streaming for " + device.getModel());
        } catch (Exception e) { Log.e(TAG, "Error stopping: " + e.getMessage()); }

        sessionBuffers.remove(id);
        pausedMap.remove(id);
        frameQueues.remove(id);
        bufferMap.remove(id);
    }

    private void startMetaWearStreaming(ImuDevice device, GyroZCallback zCallback) {
        if (device.getBoard() == null) { Log.e(TAG, "MetaWear start: board is null"); return; }
        final String deviceId = device.getMacAddress();

        Gyro gyro = device.getBoard().getModule(Gyro.class);
        gyro.configure().odr(Gyro.OutputDataRate.ODR_100_HZ).commit();
        gyro.angularVelocity().addRouteAsync(source -> source.stream((data, env) -> {
            if (Boolean.TRUE.equals(pausedMap.get(deviceId))) return;
            AngularVelocity g = data.value(AngularVelocity.class);
            float z = g.z(); long ts = System.currentTimeMillis();
            CircularBuffer cb = bufferMap.computeIfAbsent(deviceId, id -> new CircularBuffer(BUFFER_CAPACITY));
            cb.add(z, ts);
            if (zCallback != null) zCallback.onGyroZ(z);
            appendCsv(deviceId, ts, "gyro", 0f, 0f, z);
        })).continueWithTask(task -> {
            gyro.angularVelocity().start();
            gyro.start();
            kickOffAnalysisLoopIfNeeded();
            return null;
        });
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    private void startWitMotionStreaming(ImuDevice device, GyroZCallback zCallback) {
        final String deviceId = device.getMacAddress();
        final BluetoothDevice bt = device.getBluetoothDevice();

        if (bt == null) { Log.e(TAG, "WitMotion start: BluetoothDevice is null"); return; }

        frameQueues.put(deviceId, new ArrayDeque<>(64));

        BluetoothGatt gatt = bt.connectGatt(this, false, new BluetoothGattCallback() {
            private final Handler gattHandler = new Handler();
            private static final int DISCOVERY_RETRY_MS = 800;
            private static final int DISCOVERY_MAX_TRIES = 3;
            private int discoveryTries = 0;

            @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
            @Override public void onConnectionStateChange(BluetoothGatt g, int status, int newState) {
                final String id = g.getDevice().getAddress();
                Log.i(TAG, "WitMotion conn state dev=" + id + " status=" + status + " newState=" + newState);
                if (newState == BluetoothProfile.STATE_CONNECTED) {
                    try { g.requestConnectionPriority(BluetoothGatt.CONNECTION_PRIORITY_HIGH); } catch (Throwable ignored) {}
                    discoveryTries = 0;
                    gattHandler.postDelayed(() -> tryDiscoverServices(g), 300);
                } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                    gattMap.remove(id);
                    pendingGattOp.remove(id);
                    gattHandler.removeCallbacksAndMessages(null);
                    Log.w(TAG, "WitMotion disconnected (status=" + status + ")");
                }
            }

            @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
            private void tryDiscoverServices(BluetoothGatt g) {
                boolean ok = g.discoverServices();
                discoveryTries++;
                Log.i(TAG, "discoverServices() try " + discoveryTries + " -> " + ok);
                gattHandler.postDelayed(() -> {
                    if (discoveryTries <= 0) return;
                    if (discoveryTries < DISCOVERY_MAX_TRIES) tryDiscoverServices(g);
                    else Log.e(TAG, "Services discovery never returned; giving up for this device.");
                }, DISCOVERY_RETRY_MS);
            }

            @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
            @Override
            public void onServicesDiscovered(BluetoothGatt g, int status) {
                int tries = discoveryTries; discoveryTries = 0;
                final String id = g.getDevice().getAddress();
                Log.i(TAG, "onServicesDiscovered status=" + status + " (tries=" + tries + ")");

                if (status != BluetoothGatt.GATT_SUCCESS) {
                    Log.e(TAG, "Services discovery failed: " + status);
                    return;
                }

                // Dump layout (debug aid)
                for (BluetoothGattService s : g.getServices()) {
                    Log.i(TAG, "Svc: " + s.getUuid());
                    for (BluetoothGattCharacteristic c : s.getCharacteristics()) {
                        int props = c.getProperties();
                        Log.i(TAG, "  Char: " + c.getUuid() + " props=0x" + Integer.toHexString(props));
                    }
                }

                BluetoothGattCharacteristic notifyChar = pickWitNotifyOrIndicateCharacteristic(g);
                if (notifyChar == null) {
                    Log.e(TAG, "No NOTIFY/INDICATE characteristic found for WitMotion");
                    return;
                }

                try {
                    // 1) Local enable
                    boolean localOk = g.setCharacteristicNotification(notifyChar, true);
                    Log.i(TAG, "setCharacteristicNotification(" + notifyChar.getUuid() + ") -> " + localOk);

                    // 2) Write CCCD with the right value based on props
                    final int props = notifyChar.getProperties();
                    final boolean wantsIndication = (props & BluetoothGattCharacteristic.PROPERTY_INDICATE) != 0;
                    final byte[] cccdValue = wantsIndication ? BluetoothGattDescriptor.ENABLE_INDICATION_VALUE : BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE;

                    BluetoothGattDescriptor cccd = findCccd(notifyChar);
                    if (cccd != null) {
                        cccd.setValue(cccdValue);
                        pendingGattOp.put(id, GATT_OP_WRITE_CCCD);
                        boolean writeOk = g.writeDescriptor(cccd);
                        Log.i(TAG, "writeDescriptor(CCCD " + cccd.getUuid() + ", value=" +
                                (wantsIndication ? "INDICATE" : "NOTIFY") + ") -> " + writeOk);

                        if (!writeOk) {
                            // Some stacks return false but still succeed; proceed cautiously.
                            Log.w(TAG, "CCCD write returned false; proceeding cautiously");
                            maybeSendStartCommand(g);
                            kickOffAnalysisLoopIfNeeded();
                            try { g.requestMtu(185); } catch (Throwable ignored) {}
                        }
                    } else {
                        // Some clones notify without an exposed CCCD (rare). Try anyway.
                        Log.w(TAG, "No CCCD on " + notifyChar.getUuid() + " — trying without it");
                        maybeSendStartCommand(g);
                        kickOffAnalysisLoopIfNeeded();
                        try { g.requestMtu(185); } catch (Throwable ignored) {}
                    }
                } catch (Throwable t) {
                    Log.e(TAG, "Failed to enable notifications/indications", t);
                }
            }


            @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
            @Override public void onDescriptorWrite(BluetoothGatt g, BluetoothGattDescriptor d, int status) {
                final String id = g.getDevice().getAddress();
                Integer op = pendingGattOp.get(id);
                if (op != null && isCccdUuid(d.getUuid())) {
                    Log.i(TAG, "CCCD write status=" + status + " for " + d.getCharacteristic().getUuid());
                    pendingGattOp.remove(id);

                    if (SEND_WIT_START_CMD) {
                        BluetoothGattCharacteristic ctrl = findPossibleControlCharacteristic(g);
                        if (ctrl != null) {
                            ctrl.setValue(WIT_START_CMD);
                            boolean ok = g.writeCharacteristic(ctrl);
                            Log.i(TAG, "writeCharacteristic(start cmd to " + ctrl.getUuid() + ") -> " + ok);
                        }
                    }
                    kickOffAnalysisLoopIfNeeded();
                    try { g.requestMtu(185); } catch (Throwable ignored) {}
                }
            }

            @Override
            public void onCharacteristicChanged(BluetoothGatt g, BluetoothGattCharacteristic ch) {
                handleNotify(g, ch, ch.getValue());
            }

            @Override
            public void onCharacteristicChanged(BluetoothGatt g, BluetoothGattCharacteristic ch, byte[] value) {
                handleNotify(g, ch, value);
            }

            private void handleNotify(BluetoothGatt g, BluetoothGattCharacteristic ch, byte[] value) {
                final String id = g.getDevice().getAddress();
                if (Boolean.TRUE.equals(pausedMap.get(id))) return;
                if (value == null || value.length == 0) return;

                float[] zValues = parseWitMotionFrames(id, value);
                long ts = System.currentTimeMillis();
                if (zValues != null) {
                    CircularBuffer cb = bufferMap.computeIfAbsent(id, k -> new CircularBuffer(BUFFER_CAPACITY));
                    for (float z : zValues) {
                        cb.add(z, ts);
                        appendCsv(id, ts, "gyro", 0f, 0f, z);
                        if (zCallback != null) zCallback.onGyroZ(z);
                    }
                }
            }

            @Override public void onMtuChanged(BluetoothGatt g, int mtu, int status) {
                Log.i(TAG, "onMtuChanged mtu=" + mtu + " status=" + status + " dev=" + g.getDevice().getAddress());
            }
        }, BluetoothDevice.TRANSPORT_LE);

        gattMap.put(deviceId, gatt);
    }

    // ===== WitMotion helpers =====
    private boolean isCccdUuid(UUID u) {
        String s = u.toString().toLowerCase();
        return s.contains("2902");
    }

    @Nullable
    private BluetoothGattDescriptor findCccd(BluetoothGattCharacteristic ch) {
        BluetoothGattDescriptor d = ch.getDescriptor(CCCD_UUID);
        if (d != null) return d;
        for (BluetoothGattDescriptor cand : ch.getDescriptors()) {
            if (cand.getUuid().toString().toLowerCase().contains("2902")) return cand;
        }
        return null;
    }

    @Nullable private BluetoothGattCharacteristic pickWitNotifyCharacteristic(BluetoothGatt g) {
        BluetoothGattService svc = g.getService(WIT_SERVICE_UUID_9B);
        if (svc != null) {
            BluetoothGattCharacteristic ch = svc.getCharacteristic(WIT_NOTIFY_UUID_9B);
            if (ch != null) return ch;
            for (BluetoothGattCharacteristic c : svc.getCharacteristics())
                if ((c.getProperties() & BluetoothGattCharacteristic.PROPERTY_NOTIFY) != 0) return c;
        }
        svc = g.getService(WIT_SERVICE_UUID_9A);
        if (svc != null) {
            BluetoothGattCharacteristic ch = svc.getCharacteristic(WIT_NOTIFY_UUID_9A);
            if (ch != null) return ch;
            for (BluetoothGattCharacteristic c : svc.getCharacteristics())
                if ((c.getProperties() & BluetoothGattCharacteristic.PROPERTY_NOTIFY) != 0) return c;
        }
        BluetoothGattCharacteristic firstNotify = null;
        for (BluetoothGattService s : g.getServices()) {
            for (BluetoothGattCharacteristic c : s.getCharacteristics()) {
                if ((c.getProperties() & BluetoothGattCharacteristic.PROPERTY_NOTIFY) != 0) {
                    if (firstNotify == null) firstNotify = c;
                    String u = c.getUuid().toString().toLowerCase();
                    if (u.startsWith("0000ffe4-") || u.contains("ffe4")) return c;
                }
            }
        }
        return firstNotify;
    }

    @Nullable private BluetoothGattCharacteristic findPossibleControlCharacteristic(BluetoothGatt g) {
        for (BluetoothGattService s : g.getServices()) {
            for (BluetoothGattCharacteristic c : s.getCharacteristics()) {
                int props = c.getProperties();
                boolean write = (props & BluetoothGattCharacteristic.PROPERTY_WRITE) != 0
                        || (props & BluetoothGattCharacteristic.PROPERTY_WRITE_NO_RESPONSE) != 0;
                if (!write) continue;
                String u = c.getUuid().toString().toLowerCase();
                if (u.contains("ffe9") || u.contains("ffe2")) return c;
            }
        }
        return null;
    }

    /** Assemble WT901 11-byte frames and return parsed Z-gyro values (°/s). */
    private float[] parseWitMotionFrames(String deviceId, byte[] incoming) {
        Deque<Byte> q = frameQueues.get(deviceId); if (q == null) return null;
        for (byte b : incoming) q.addLast(b);
        List<Float> zs = new ArrayList<>();
        while (q.size() >= 11) {
            if ((q.peekFirst() & 0xFF) != 0x55) { q.removeFirst(); continue; }
            byte[] peek = new byte[11]; int i = 0; for (Byte val : q) { if (i >= 11) break; peek[i++] = val; }
            if (i < 11) break;
            if ((peek[1] & 0xFF) == 0x52) {
                int sum = 0; for (int k = 0; k < 10; k++) sum += (peek[k] & 0xFF);
                if (((sum & 0xFF) != (peek[10] & 0xFF))) { q.removeFirst(); continue; }
                for (int k = 0; k < 11; k++) q.removeFirst();
                short gzRaw = ByteBuffer.wrap(new byte[]{peek[6], peek[7]}).order(ByteOrder.LITTLE_ENDIAN).getShort();
                zs.add(gzRaw / 32768.0f * 2000.0f);
            } else { q.removeFirst(); }
        }
        if (zs.isEmpty()) return null;
        float[] out = new float[zs.size()]; for (int k = 0; k < zs.size(); k++) out[k] = zs.get(k); return out;
    }

    private final List<DataPoint> leftZ = new ArrayList<>();
    private final List<DataPoint> rightZ = new ArrayList<>();

    private final Runnable analysisRunnable = new Runnable() {
        @Override public void run() {
            leftZ.clear(); rightZ.clear();
            List<DataPoint> allLeft = new ArrayList<>();
            List<DataPoint> allRight = new ArrayList<>();

            for (Map.Entry<String, CircularBuffer> entry : bufferMap.entrySet()) {
                final String deviceId = entry.getKey();
                final CircularBuffer buffer = entry.getValue();
                final String side = deviceToSideMap.get(deviceId);
                CircularBuffer.Snapshot snap = buffer.snapshot();
                float[] zVals = snap.z(); long[] tVals = snap.t(); int len = zVals.length; if (len == 0) continue;
                List<DataPoint> pts = new ArrayList<>(len);
                for (int i = 0; i < len; i++) pts.add(new DataPoint(zVals[i], tVals[i]));
                if ("left".equals(side))  allLeft.addAll(pts);
                if ("right".equals(side)) allRight.addAll(pts);
            }

            if (allLeft.isEmpty() || allRight.isEmpty()) { analysisHandler.postDelayed(this, ANALYSIS_INTERVAL_MS); return; }

            long latestLeftTs  = allLeft.get(allLeft.size() - 1).timestamp;
            long latestRightTs = allRight.get(allRight.size() - 1).timestamp;
            long endTs = Math.min(latestLeftTs, latestRightTs);
            long startTs = endTs - WINDOW_MS;

            if (endTs - lastProcessedEndTs < HOP_MS) { analysisHandler.postDelayed(this, ANALYSIS_INTERVAL_MS); return; }

            for (DataPoint p : allLeft)  if (p.timestamp >= startTs && p.timestamp <= endTs) leftZ.add(p);
            for (DataPoint p : allRight) if (p.timestamp >= startTs && p.timestamp <= endTs) rightZ.add(p);

            if (leftZ.size() < WINDOW_SAMPLES * 0.8f || rightZ.size() < WINDOW_SAMPLES * 0.8f) { analysisHandler.postDelayed(this, ANALYSIS_INTERVAL_MS); return; }

            long now = SystemClock.elapsedRealtime();
            if (now - lastInferenceTs < INFERENCE_COOLDOWN_MS) { analysisHandler.postDelayed(this, ANALYSIS_INTERVAL_MS); return; }
            lastInferenceTs = now; lastProcessedEndTs = endTs;

            GaitWindowResult result = featureExtraction(leftZ, rightZ);
            modelPrediction(result);
            analysisHandler.postDelayed(this, ANALYSIS_INTERVAL_MS);
        }
    };

    private void kickOffAnalysisLoopIfNeeded() {
        if (!analysisStarted) {
            analysisStarted = true;
            Log.d(TAG, "Starting analysis loop");
            analysisHandler.postDelayed(analysisRunnable, ANALYSIS_INTERVAL_MS);
        }
    }

    // ===== Model + alert =====
    private void modelPrediction(GaitWindowResult r) {
        Log.d(TAG, "==== Gait Values ====");
        Log.d(TAG, "Left Stance:  " + r.getLeftStance()  + "s");
        Log.d(TAG, "Left Swing :  " + r.getLeftSwing()   + "s");
        Log.d(TAG, "Right Stance: " + r.getRightStance() + "s");
        Log.d(TAG, "Right Swing : " + r.getRightSwing()  + "s");

        if (!r.gaitWindowValid() || predictor == null) {
            Log.w(TAG, "Predictor not initialized or invalid window, skipping prediction");
            return;
        }

        float[] modelInput = new float[]{ r.getRightStance(), r.getLeftStance(), r.getRightSwing(), r.getLeftSwing() };

        try {
            int prediction = predictor.predict(modelInput);
            Log.d(TAG, "Predicted gait status: " + prediction);
            if (prediction == 1) {
                long now = SystemClock.elapsedRealtime();
                if (now - lastAlertTs >= ALERT_COOLDOWN_MS) { lastAlertTs = now; showAlarmAlert(); }
            }
        }
        catch (Exception e) {
            Log.e(TAG, "Prediction failed: " + e.getMessage());
        }


        Log.d(TAG, "========================");
    }

    private void showAlarmAlert() {
        Notification notification = new NotificationCompat.Builder(this, ALERT_CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setContentTitle("Gait Alert")
                .setContentText("Abnormal gait detected — please adjust.")
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setCategory(NotificationCompat.CATEGORY_ALARM)
                .setAutoCancel(true)
                .build();
        int id = (int) System.currentTimeMillis();
        notificationManager.notify(id, notification);
        startAlarmTone();
        analysisHandler.postDelayed(this::stopAlarmTone, ALERT_DURATION_MS);
    }

    private void startAlarmTone() {
        try {
            if (alarmRingtone != null && alarmRingtone.isPlaying()) return;
            Uri uri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM);
            if (uri == null) uri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION);
            alarmRingtone = RingtoneManager.getRingtone(getApplicationContext(), uri);
            if (alarmRingtone != null) {
                alarmRingtone.setAudioAttributes(new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ALARM)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build());
                alarmRingtone.play();
            }
        } catch (Throwable t) { Log.w(TAG, "Failed to play alarm tone", t); }
    }

    private void stopAlarmTone() {
        try {
            if (alarmRingtone != null && alarmRingtone.isPlaying())
                alarmRingtone.stop();
        }
        catch (Throwable ignored) {}
    }


    private void appendCsv(String deviceId, long ts, String type, float x, float y, float z) {
        StringBuilder sb = sessionBuffers.computeIfAbsent(deviceId, k -> new StringBuilder(8 * 1024));
        sb.append(ts).append(',').append(type).append(',').append(x).append(',').append(y).append(',').append(z).append('\n');
    }

    @RequiresApi(api = Build.VERSION_CODES.VANILLA_ICE_CREAM)
    private void exportToCSV(String deviceId, StringBuilder buffer) {
        if (buffer == null || buffer.isEmpty()) return;
        File dir = new File(getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), "IMU_Logs");
        if (!dir.exists()) //noinspection ResultOfMethodCallIgnored
            dir.mkdirs();
        File file = new File(dir, deviceId + "_session.csv");
        try (FileWriter writer = new FileWriter(file)) {
            writer.write("timestamp,type,x,y,z\n");
            writer.write(buffer.toString());
            Log.i(TAG, "Saved session to: " + file.getAbsolutePath());
        } catch (IOException e) { Log.e(TAG, "CSV export failed: " + e.getMessage()); }
    }

    @Nullable
    private BluetoothGattCharacteristic pickWitNotifyOrIndicateCharacteristic(BluetoothGatt g) {
        // Prefer the common FFE4 char under FFE5 service (both 9b/9a bases)
        BluetoothGattService svc = g.getService(WIT_SERVICE_UUID_9B);
        if (svc == null) svc = g.getService(WIT_SERVICE_UUID_9A);

        if (svc != null) {
            BluetoothGattCharacteristic ch =
                    svc.getCharacteristic(WIT_NOTIFY_UUID_9B) != null ? svc.getCharacteristic(WIT_NOTIFY_UUID_9B)
                            : svc.getCharacteristic(WIT_NOTIFY_UUID_9A);
            if (ch != null) return ch;
            for (BluetoothGattCharacteristic c : svc.getCharacteristics()) {
                int p = c.getProperties();
                if ((p & (BluetoothGattCharacteristic.PROPERTY_NOTIFY | BluetoothGattCharacteristic.PROPERTY_INDICATE)) != 0) {
                    return c;
                }
            }
        }

        // Fallback: first char anywhere that supports notify/indicate; bias to “ffe4”
        BluetoothGattCharacteristic first = null;
        for (BluetoothGattService s : g.getServices()) {
            for (BluetoothGattCharacteristic c : s.getCharacteristics()) {
                int p = c.getProperties();
                if ((p & (BluetoothGattCharacteristic.PROPERTY_NOTIFY | BluetoothGattCharacteristic.PROPERTY_INDICATE)) != 0) {
                    if (first == null) first = c;
                    String u = c.getUuid().toString().toLowerCase();
                    if (u.contains("ffe4")) return c;
                }
            }
        }
        return first;
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    private void maybeSendStartCommand(BluetoothGatt g) {
        if (!SEND_WIT_START_CMD) return;
        BluetoothGattCharacteristic ctrl = findPossibleControlCharacteristic(g); // you already have this
        if (ctrl == null) {
            Log.w(TAG, "No control characteristic (FFE9/FFE2) found; device may stream by default");
            return;
        }
        try {
            ctrl.setWriteType(BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT); // with response
            for (byte[] cmd : WIT_START_CMDS) {
                ctrl.setValue(cmd);
                boolean ok = g.writeCharacteristic(ctrl);
                Log.i(TAG, "writeCharacteristic(" + ctrl.getUuid() + ", len=" + cmd.length + ") -> " + ok);
                if (ok) break;
            }
        } catch (Throwable t) {
            Log.w(TAG, "Failed to send start command", t);
        }
    }
}
