package com.example.gaitrehabapp;

import android.Manifest;
import android.content.Intent;
import android.os.Bundle;
import androidx.annotation.RequiresPermission;
import androidx.appcompat.app.AppCompatActivity;


public class MainActivity extends AppCompatActivity {

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        findViewById(R.id.scanButton).setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, DeviceScanActivity.class);
            startActivity(intent);
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }
}
