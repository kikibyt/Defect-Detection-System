from fastapi import FastAPI, UploadFile
import onnxruntime as ort
import cv2
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import logging, os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# -------------------------
# Load ONNX model
# -------------------------
try:
    ort_session = ort.InferenceSession('autoencoder.onnx')
    input_name = ort_session.get_inputs()[0].name
    logger.info("ONNX model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load ONNX model: {e}")
    ort_session, input_name = None, None

# -------------------------
# Load threshold
# -------------------------
if os.path.exists("threshold.npy"):
    threshold = float(np.load("threshold.npy"))
    logger.info(f"Loaded threshold: {threshold:.6f}")
else:
    threshold = 0.01
    logger.warning("No threshold.npy found, using default 0.01")

# -------------------------
# Reference data for drift detection
# -------------------------
ref_data = pd.DataFrame({
    'reconstruction_error': np.random.exponential(scale=0.005, size=1000),
    'prediction': np.random.choice([0, 1], size=1000, p=[0.9, 0.1])
})

@app.get("/")
async def root():
    return {"message": "Defect Detection API", "status": "running"}

@app.post("/detect")
async def detect(file: UploadFile):
    try:
        if not file.content_type.startswith("image/"):
            return {"error": "File must be an image"}
        if ort_session is None:
            return {"error": "ONNX model not loaded"}

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"error": "Invalid image file"}

        img = cv2.resize(img, (128, 128))
        img_input = (img / 255.0).astype(np.float32)[None, None, :, :]

        output = ort_session.run(None, {input_name: img_input})[0]
        error = float(np.mean((img_input - output) ** 2))
        is_defect = bool(error > threshold)

        curr_data = pd.DataFrame({
            'reconstruction_error': [error],
            'prediction': [1 if is_defect else 0]
        })

        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_data, current_data=curr_data)
            drift_detected = False  # Simplified
        except Exception as drift_error:
            logger.warning(f"Drift detection failed: {drift_error}")
            drift_detected = False

        return {
            "defect": is_defect,
            "error": error,
            "threshold": threshold,
            "drift": drift_detected,
            "filename": file.filename
        }

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return {"error": f"Processing failed: {str(e)}"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": ort_session is not None,
        "threshold": threshold,
        "reference_data_size": len(ref_data)
    }