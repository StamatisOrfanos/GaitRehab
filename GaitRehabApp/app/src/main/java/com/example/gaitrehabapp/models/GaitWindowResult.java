package com.example.gaitrehabapp.models;

public class GaitWindowResult {
    public float leftStance;
    public float leftSwing;
    public float rightStance;
    public float rightSwing;

    public GaitWindowResult() {}

    public GaitWindowResult(float leftStance, float leftSwing, float rightStance, float rightSwing) {
        this.leftStance = leftStance;
        this.leftSwing = leftSwing;
        this.rightStance = rightStance;
        this.rightSwing = rightSwing;
    }

    public float getLeftSwing() {
        return leftSwing;
    }

    public void setLeftSwing(float leftSwing) {
        this.leftSwing = leftSwing;
    }

    public float getRightStance() {
        return rightStance;
    }

    public void setRightStance(float rightStance) {
        this.rightStance = rightStance;
    }

    public float getLeftStance() {
        return leftStance;
    }

    public void setLeftStance(float leftStance) {
        this.leftStance = leftStance;
    }

    public float getRightSwing() {
        return rightSwing;
    }

    public void setRightSwing(float rightSwing) {
        this.rightSwing = rightSwing;
    }

    @Override
    public String toString() {
        return "GaitWindowResult{" +
                "leftStance=" + leftStance +
                ", leftSwing=" + leftSwing +
                ", rightStance=" + rightStance +
                ", rightSwing=" + rightSwing +
                '}';
    }

    public boolean gaitWindowValid() {
        return !(Float.isNaN(leftStance) || Float.isNaN(leftSwing) || Float.isNaN(rightStance) || Float.isNaN(rightSwing));
    }
}
