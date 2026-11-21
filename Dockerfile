
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scikit-learn==1.5.2 \
    joblib==1.4.2 \
    fastapi==0.115.6 \
    uvicorn==0.34.0

COPY app.py .
COPY models/raisin_model.joblib models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
