import torch
import torch.onnx
import numpy as np
import os
from model import Autoencoder

# -------------------------
# Load trained model
# -------------------------
model = Autoencoder()
model.load_state_dict(torch.load("autoencoder_best.pth", map_location="cpu"))
model.eval()

# Dummy input for export
dummy_input = torch.randn(1, 1, 128, 128)

# -------------------------
# Export to ONNX
# -------------------------
onnx_path = "autoencoder.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print(f"✅ Model exported to {onnx_path}")

# -------------------------
# Ensure threshold.npy is present
# -------------------------
if os.path.exists("threshold.npy"):
    print("✅ threshold.npy found, reusing existing threshold")
else:
    default_threshold = 0.01
    np.save("threshold.npy", default_threshold)
    print(f"⚠️ No threshold.npy found. Saved default threshold = {default_threshold}")
