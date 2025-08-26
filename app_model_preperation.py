import joblib
import numpy as np
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# Load saved pipeline
pipeline = joblib.load("models/detection/xgboost_moderate_conf.pkl")

# Separate components
scaler = pipeline.named_steps['standardscaler']
model = pipeline.named_steps['xgbclassifier']

# Explicitly save as float32 to match the Android app's expectation
np.save("scaler_mean.npy", scaler.mean_.astype(np.float32))
np.save("scaler_scale.npy", scaler.scale_.astype(np.float32))

# Convert XGBoost model to ONNX
initial_type = [('float_input', FloatTensorType([None, model.n_features_in_]))]
onnx_model = convert_xgboost(model, initial_types=initial_type)

# Save ONNX model
with open("xgboost_moderate_conf.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())