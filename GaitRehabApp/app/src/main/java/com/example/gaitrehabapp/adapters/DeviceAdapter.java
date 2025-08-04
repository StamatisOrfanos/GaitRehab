package com.example.gaitrehabapp.adapters;

import android.annotation.SuppressLint;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.example.gaitrehabapp.R;
import com.example.gaitrehabapp.models.ImuDevice;

import java.util.ArrayList;
import java.util.List;

public class DeviceAdapter extends RecyclerView.Adapter<DeviceAdapter.DeviceViewHolder> {

    private final List<ImuDevice> devices = new ArrayList<>();
    private final List<ImuDevice> selectedDevices = new ArrayList<>();
    private static final int MAX_SELECTION = 2;

    @NonNull
    @Override
    public DeviceViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View itemView = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.device_item, parent, false);
        return new DeviceViewHolder(itemView);
    }

    @Override
    public void onBindViewHolder(@NonNull DeviceViewHolder holder, int position) {
        ImuDevice device = devices.get(position);
        holder.deviceName.setText(device.getModel() != null ? device.getModel() : "Unnamed Device");
        holder.deviceMac.setText(device.getMacAddress());

        holder.deviceCheckbox.setOnCheckedChangeListener(null);
        holder.deviceCheckbox.setChecked(selectedDevices.contains(device));

        holder.deviceCheckbox.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked) {
                if (selectedDevices.size() < MAX_SELECTION) {
                    selectedDevices.add(device);
                } else {
                    buttonView.setChecked(false);
                }
            } else {
                selectedDevices.remove(device);
            }
        });

        holder.itemView.setOnClickListener(v -> holder.deviceCheckbox.performClick());
    }

    @Override
    public int getItemCount() {
        return devices.size();
    }

    @SuppressLint("NotifyDataSetChanged")
    public void updateDevices(List<ImuDevice> newDevices) {
        devices.clear();
        devices.addAll(newDevices);
        selectedDevices.clear();
        notifyDataSetChanged();
    }

    public List<ImuDevice> getSelectedDevices() {
        return new ArrayList<>(selectedDevices);
    }

    static class DeviceViewHolder extends RecyclerView.ViewHolder {
        TextView deviceName, deviceMac;
        CheckBox deviceCheckbox;

        public DeviceViewHolder(@NonNull View itemView) {
            super(itemView);
            deviceName = itemView.findViewById(R.id.deviceName);
            deviceMac = itemView.findViewById(R.id.deviceMac);
            deviceCheckbox = itemView.findViewById(R.id.deviceCheckbox);
        }
    }
}
