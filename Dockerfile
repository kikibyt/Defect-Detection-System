FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "80"]

