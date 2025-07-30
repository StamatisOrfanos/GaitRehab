package com.example.gaitrehabapp;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.example.gaitrehabapp.models.ImuDevice;
import java.util.ArrayList;
import java.util.List;

public class DeviceAdapter extends RecyclerView.Adapter<DeviceAdapter.DeviceViewHolder> {

    private final List<ImuDevice> devices;
    private final List<ImuDevice> selected = new ArrayList<>();

    public DeviceAdapter(List<ImuDevice> devices) {
        this.devices = devices;
    }

    public List<ImuDevice> getSelectedDevices() {
        return selected;
    }

    @NonNull
    @Override
    public DeviceViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(parent.getContext()).inflate(R.layout.device_item, parent, false);
        return new DeviceViewHolder(v);
    }

    @Override
    public void onBindViewHolder(@NonNull DeviceViewHolder holder, int position) {
        ImuDevice device = devices.get(position);
        holder.name.setText(device.getName() != null ? device.getName() : "Unknown");
        holder.mac.setText(device.getMacAddress());

        holder.checkbox.setOnCheckedChangeListener(null);
        holder.checkbox.setChecked(selected.contains(device));

        holder.checkbox.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked && selected.size() < 2) {
                selected.add(device);
            } else if (!isChecked) {
                selected.remove(device);
            } else {
                buttonView.setChecked(false);
            }
        });
    }

    @Override
    public int getItemCount() {
        return devices.size();
    }

    static class DeviceViewHolder extends RecyclerView.ViewHolder {
        TextView name, mac;
        CheckBox checkbox;
        DeviceViewHolder(@NonNull View itemView) {
            super(itemView);
            name = itemView.findViewById(R.id.deviceName);
            checkbox = itemView.findViewById(R.id.deviceCheckbox);
        }
    }
}
