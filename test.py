import os
import pandas as pd
import numpy as np
from data_preprocessing import aggregate_features


if __name__ == "__main__":
    # === Change these ===
    output_path = "final_dataset.csv"

    # Example usage for healthy + (optionally) stroke folders
    healthy_df = aggregate_features(base_dir="Data/Healthy", label=0)

    # Uncomment if you have stroke subjects:
    stroke_df = aggregate_features(base_dir="Data/Stroke", label=1)
    
    # Combine both datasets
    full_df = pd.concat([healthy_df, stroke_df], ignore_index=True)
    full_df.to_csv(output_path, index=False)

    # Only healthy for now:
    full_df.to_csv(output_path, index=False)
    print(f" Saved dataset with shape {full_df.shape} to {output_path}")
