Defect Detection MVP
Pipeline for detecting defects in images using an autoencoder.
Setup

Download MVTec AD dataset (bottle category) to data/mvtec/bottle.
Install dependencies: pip install -r requirements.txt.
Train: python src/train.py.
Export to ONNX: python src/export_onnx.py.
Run API: uvicorn src.app:app --reload.
Build Docker: docker build -t defect-api . && docker run -p 8000:80 defect-api.

Endpoint

POST /detect: Upload image, returns defect status, error, and drift.
docker build -t defect-api .
docker run -p 8000:80 defect-api
