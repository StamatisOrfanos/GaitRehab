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
import java.nio.charset.StandardCharsets;
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
    private static final boolean SEND_WIT_START_CMD = true;
    // Binders
    public class LocalBinder extends Binder { public ImuStreamService getService() { return ImuStreamService.this; } }
    @Nullable @Override public IBinder onBind(Intent intent) { return binder; }
    private final IBinder binder = new LocalBinder();
    //
    private final Handler analysisHandler = new Handler();
    private final Map<String, CircularBuffer> bufferMap = new HashMap<>();
    private final Map<String, Boolean> pausedMap = new HashMap<>();
    private final Map<String, String> deviceToSideMap = new HashMap<>();
    private final Map<String, StringBuilder> sessionBuffers = new HashMap<>();
    private final Map<String, long[]> idCounts = new HashMap<>();
    private final Map<String, Long>   last62Seen = new HashMap<>();
    private static final byte[][] WIT_START_CMDS = new byte[][]{
            {(byte)0xFF, (byte)0xAA, 0x00, 0x00, 0x00},                 // factory unlock
            {(byte)0xFF, (byte)0xAA, 0x52, 0x02, 0x00},                 // stream mode (typical)
            {(byte)0xFF, (byte)0xAA, 0x69, (byte)0x88, (byte)0xB5},     // config unlock (alt firmwares)
            {(byte)0xFF, (byte)0xAA, 0x02, 0x07, 0x00},                 // enable ACC(1)+GYR(2)+ANGLE(4)
            {(byte)0xFF, (byte)0xAA, 0x02, 0x06, 0x00},                 // enable ACC+GYR only
            {(byte)0xFF, (byte)0xAA, 0x03, 0x0A, 0x00},                 // ODR ≈ 100 Hz (varies by model)
            "AT+SENSOR=ACC+GYRO".getBytes(StandardCharsets.US_ASCII),
            "AT+ODR=100".getBytes(StandardCharsets.US_ASCII)
    };

    private final List<DataPoint> leftZ = new ArrayList<>();
    private final List<DataPoint> rightZ = new ArrayList<>();

    private final Map<String, Long> lastRxTs = new HashMap<>();
    private boolean analysisStarted = false;
    private long lastInferenceTs = 0L;
    private long lastAlertTs = 0L;
    private long lastProcessedEndTs = 0L;
    private ModelPredictor predictor;
    private NotificationManager notificationManager;
    private Ringtone alarmRingtone;
    private final Map<String, BluetoothGatt> gattMap = new HashMap<>();
    private final Map<String, Deque<Byte>> frameQueues = new HashMap<>();


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
            metaWearStreaming(device, zCallback);
        } else if (device.getDeviceType() == DeviceType.WITMOTION) {
            witMotionStreaming(device, zCallback);
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


    // Feature Generation, Model Prediction and Alert System
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


    // MetaWear Functionality
    private void metaWearStreaming(ImuDevice device, GyroZCallback zCallback) {
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
        })).continueWithTask(task -> {
            gyro.angularVelocity().start();
            gyro.start();
            kickOffAnalysisLoopIfNeeded();
            return null;
        });
    }
    private void kickOffAnalysisLoopIfNeeded() {
        if (!analysisStarted) {
            analysisStarted = true;
            Log.d(TAG, "Starting analysis loop");
            analysisHandler.postDelayed(analysisRunnable, ANALYSIS_INTERVAL_MS);
        }
    }


    // WitMotion Functionality
    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    private void witMotionStreaming(ImuDevice device, GyroZCallback zCallback) {
        final String deviceId = device.getMacAddress();
        final BluetoothDevice bt = device.getBluetoothDevice();
        if (bt == null) { Log.e(TAG, "WitMotion start: BluetoothDevice is null"); return; }

        frameQueues.put(deviceId, new ArrayDeque<>(64));

        BluetoothGatt gatt = bt.connectGatt(this, false, new BluetoothGattCallback() {
            private final Handler gattHandler = new Handler();
            private static final int DISCOVERY_RETRY_MS = 800;
            private static final int DISCOVERY_MAX_TRIES = 3;
            private int discoveryTries = 0;
            private boolean servicesReady = false;

            @Override @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
            public void onConnectionStateChange(BluetoothGatt g, int status, int newState) {
                final String id = g.getDevice().getAddress();
                Log.i(TAG, "WitMotion conn state dev=" + id + " status=" + status + " newState=" + newState);
                if (newState == BluetoothProfile.STATE_CONNECTED) {
                    gattMap.put(id, g);
                    servicesReady = false;
                    try { g.requestConnectionPriority(BluetoothGatt.CONNECTION_PRIORITY_HIGH); } catch (Throwable ignored) {}
                    discoveryTries = 0;
                    gattHandler.postDelayed(() -> tryDiscoverServices(g), 300);
                } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                    servicesReady = false;
                    gattMap.remove(id);
                    gattHandler.removeCallbacksAndMessages(null);
                    Log.w(TAG, "WitMotion disconnected (status=" + status + ")");
                }
            }

            @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
            private void tryDiscoverServices(BluetoothGatt g) {
                if (servicesReady) return;
                boolean ok = g.discoverServices();
                discoveryTries++;
                Log.i(TAG, "discoverServices() try " + discoveryTries + " -> " + ok);
                gattHandler.postDelayed(() -> {
                    if (servicesReady) return;
                    if (discoveryTries < DISCOVERY_MAX_TRIES) tryDiscoverServices(g);
                    else Log.e(TAG, "Services discovery never returned; giving up for this device.");
                }, DISCOVERY_RETRY_MS);
            }

            @Override @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
            public void onServicesDiscovered(BluetoothGatt g, int status) {
                if (status != BluetoothGatt.GATT_SUCCESS) { Log.e(TAG, "Services discovery failed: " + status); return; }
                servicesReady = true;
                discoveryTries = 0;
                enableAllNotifies(g);
                try { g.requestMtu(185); } catch (Throwable ignored) {}

                gattHandler.postDelayed(() -> {
                    List<BluetoothGattCharacteristic> writers = allWriteChars(g);
                    if (writers.isEmpty()) {
                        Log.w(TAG, "No writeable chars found (FFE9/FFE2/FFF9/FFF2/etc). Some firmwares still stream by default.");
                    }
                    sendInitBurst(g, writers);
                    armNoDataWatchdog(g.getDevice().getAddress(), g, writers);
                }, 300);
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
                if (value == null || value.length == 0 || Boolean.TRUE.equals(pausedMap.get(id))) return;

                lastRxTs.put(id, SystemClock.elapsedRealtime());

                if (value.length >= 2) {
                    Log.d(TAG, "NTF " + id + " " + ch.getUuid() +
                            " len=" + value.length + " b0=0x" + Integer.toHexString(value[0] & 0xFF) +
                            " b1=0x" + Integer.toHexString(value[1] & 0xFF));
                }

                float[] zValues = parseWitMotionFrames(id, value);
                long ts = System.currentTimeMillis();
                if (zValues != null) {
                    CircularBuffer cb = bufferMap.computeIfAbsent(id, k -> new CircularBuffer(BUFFER_CAPACITY));
                    for (float z : zValues) {
                        cb.add(z, ts);
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
    private void bumpIdCount(String devId, int fid) {
        long[] arr = idCounts.computeIfAbsent(devId, k -> new long[0x70]);
        if (fid >= 0 && fid < arr.length) arr[fid]++;
        if (fid == 0x62) last62Seen.put(devId, SystemClock.elapsedRealtime());
        long total = arr[0x61] + arr[0x62] + arr[0x63];
        if (total % 200 == 0) {
            Log.i(TAG, "Frame mix dev=" + devId + " 61=" + arr[0x61] + " 62=" + arr[0x62] + " 63=" + arr[0x63]);
        }
    }
    private float[] parseWitMotionFrames(String deviceId, byte[] incoming) {
        Deque<Byte> q = frameQueues.get(deviceId);
        if (q == null) return null;

        for (byte b : incoming) {
            if (q.size() > 1024) q.pollFirst();
            q.addLast(b);
        }

        List<Float> zs = new ArrayList<>();
        while (true) {
            if (q.size() < 2) break;
            if ((q.peekFirst() & 0xFF) != 0x55) { q.removeFirst(); continue; }

            // Peek 2nd byte (frame id)
            Byte[] peek2 = q.toArray(new Byte[Math.min(q.size(), 2)]);
            if (peek2.length < 2) break;
            int id = peek2[1] & 0xFF;

            // track mix for diagnostics
            bumpIdCount(deviceId, id);            // <-- NEW

            // 11-byte legacy
            if (id >= 0x51 && id <= 0x5F) {
                if (q.size() < 11) break;
                byte[] frame = new byte[11];
                for (int i = 0; i < 11; i++) frame[i] = q.removeFirst();
                int sum = 0; for (int i = 0; i < 10; i++) sum += (frame[i] & 0xFF);
                if (((sum & 0xFF) != (frame[10] & 0xFF))) continue;
                if (id == 0x52) {
                    short gzRaw = (short)((frame[7] & 0xFF) << 8 | (frame[6] & 0xFF));
                    float gz = gzRaw / 32768.0f * 2000.0f;
                    zs.add(gz);
                }
                continue;
            }

            // 20-byte new format: 0x61(acc), 0x62(gyro), 0x63(angle)
            if (id >= 0x61 && id <= 0x6F) {
                final int FRAME_LEN = 20;
                if (q.size() < FRAME_LEN) break;
                byte[] frame = new byte[FRAME_LEN];
                for (int i = 0; i < FRAME_LEN; i++) frame[i] = q.removeFirst();
                int sum = 0; for (int i = 0; i < FRAME_LEN - 1; i++) sum += (frame[i] & 0xFF);
                if (((sum & 0xFF) != (frame[FRAME_LEN - 1] & 0xFF))) continue;

                if (id == 0x62) { // gyro
                    short gzRaw = (short)((frame[7] & 0xFF) << 8 | (frame[6] & 0xFF));
                    float gz = gzRaw / 32768.0f * 2000.0f;
                    zs.add(gz);
                } else if (id == 0x61) { // OPTIONAL: quick sanity log for acc Z
                    // short azRaw = (short)((frame[5] & 0xFF) << 8 | (frame[4] & 0xFF));
                    // float az_g = azRaw / 32768f * 16f;
                    // Log.v(TAG, "ACC Z(g): " + az_g);
                }
                continue;
            }

            q.removeFirst(); // unknown id -> resync
        }

        if (zs.isEmpty()) return null;
        float[] out = new float[zs.size()];
        for (int i = 0; i < zs.size(); i++) out[i] = zs.get(i);
        return out;
    }
    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    private List<BluetoothGattCharacteristic> enableAllNotifies(BluetoothGatt g) {
        List<BluetoothGattCharacteristic> enabled = new ArrayList<>();
        for (BluetoothGattService s : g.getServices()) {
            for (BluetoothGattCharacteristic c : s.getCharacteristics()) {
                int p = c.getProperties();
                boolean canNotify   = (p & BluetoothGattCharacteristic.PROPERTY_NOTIFY) != 0;
                boolean canIndicate = (p & BluetoothGattCharacteristic.PROPERTY_INDICATE) != 0;
                if (!canNotify && !canIndicate) continue;

                boolean localOk = g.setCharacteristicNotification(c, true);
                Log.i(TAG, "setCharacteristicNotification(" + c.getUuid() + ") -> " + localOk);

                BluetoothGattDescriptor cccd = findCccdFlexible(c);
                if (cccd != null) {
                    byte[] v = canIndicate ? BluetoothGattDescriptor.ENABLE_INDICATION_VALUE
                            : BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE;
                    cccd.setValue(v);
                    boolean wrote = g.writeDescriptor(cccd);
                    Log.i(TAG, "writeDescriptor(CCCD " + cccd.getUuid() + ", " + (canIndicate?"INDICATE":"NOTIFY") + ") -> " + wrote);

                    // Retry once if platform returned false immediately
                    if (!wrote) {
                        analysisHandler.postDelayed(() -> {
                            try {
                                cccd.setValue(v);
                                boolean retry = g.writeDescriptor(cccd);
                                Log.i(TAG, "CCCD retry -> " + retry);
                            } catch (Throwable t) {
                                Log.w(TAG, "CCCD retry failed", t);
                            }
                        }, 200);
                    }
                } else {
                    Log.w(TAG, "No CCCD for " + c.getUuid() + " (device may still notify)");
                }
                enabled.add(c);
            }
        }
        Log.i(TAG, "Enabled notifications on " + enabled.size() + " characteristics");
        return enabled;
    }
    private List<BluetoothGattCharacteristic> allWriteChars(BluetoothGatt g) {
        List<BluetoothGattCharacteristic> writers = new ArrayList<>();
        List<BluetoothGattCharacteristic> biased  = new ArrayList<>();
        for (BluetoothGattService s : g.getServices()) {
            for (BluetoothGattCharacteristic c : s.getCharacteristics()) {
                int p = c.getProperties();
                boolean canWrite = (p & BluetoothGattCharacteristic.PROPERTY_WRITE) != 0
                        || (p & BluetoothGattCharacteristic.PROPERTY_WRITE_NO_RESPONSE) != 0;
                if (!canWrite) continue;
                writers.add(c);
                String u = c.getUuid().toString().toLowerCase();
                if (u.contains("ffe9") || u.contains("ffe2") || u.contains("fff9") || u.contains("fff2")) {
                    biased.add(c);
                }
            }
        }
        return !biased.isEmpty() ? biased : writers;
    }
    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    private void sendInitBurst(BluetoothGatt g, List<BluetoothGattCharacteristic> writers) {
        if (!SEND_WIT_START_CMD || writers.isEmpty()) return;
        byte[][] cmds = WIT_START_CMDS;
        final int[] idx = {0};
        Runnable writeNext = new Runnable() {
            @Override @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
            public void run() {
                if (idx[0] >= writers.size()*cmds.length) return;
                int wIndex = idx[0] / cmds.length;
                int cIndex = idx[0] % cmds.length;
                BluetoothGattCharacteristic ctrl = writers.get(wIndex);
                try {
                    ctrl.setWriteType(BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT);
                    ctrl.setValue(cmds[cIndex]);
                    boolean ok = g.writeCharacteristic(ctrl);
                    Log.i(TAG, "initBurst write -> char=" + ctrl.getUuid() + " cmd#" + cIndex + " ok=" + ok + " len=" + cmds[cIndex].length);
                } catch (Throwable t) {
                    Log.w(TAG, "initBurst write failed", t);
                }
                idx[0]++;
                analysisHandler.postDelayed(this, 120);
            }
        };
        analysisHandler.post(writeNext);
    }
    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    private void armNoDataWatchdog(String id, BluetoothGatt g, List<BluetoothGattCharacteristic> writers) {
        lastRxTs.put(id, SystemClock.elapsedRealtime());
        Runnable tick = new Runnable() {
            @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
            @Override public void run() {
                long now = SystemClock.elapsedRealtime();
                long lastAny = lastRxTs.getOrDefault(id, 0L);
                if (now - lastAny > 3000) {
                    Log.w(TAG, "No notifications received in 3s; retrying init burst");
                    sendInitBurst(g, writers);
                    lastRxTs.put(id, now); // avoid spamming
                }

                long lastGyro = last62Seen.getOrDefault(id, 0L);
                if (lastGyro == 0L || now - lastGyro > 5000) {
                    Log.w(TAG, "No 0x62 (gyro) seen for 5s; re-sending init burst");
                    sendInitBurst(g, writers);
                }

                analysisHandler.postDelayed(this, 3200); // keep checking
            }
        };
        analysisHandler.postDelayed(tick, 3200);
    }
    @Nullable
    private BluetoothGattDescriptor findCccdFlexible(BluetoothGattCharacteristic ch) {
        BluetoothGattDescriptor d = ch.getDescriptor(UUID.fromString("00002902-0000-1000-8000-00805f9b34fb"));
        if (d != null) return d;
        for (BluetoothGattDescriptor cand : ch.getDescriptors()) {
            String u = cand.getUuid().toString().toLowerCase();
            if (u.contains("2902")) return cand;
        }
        return null;
    }
}
