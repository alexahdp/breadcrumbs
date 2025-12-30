# Production Deployment

## Table of Contents

1. [From Notebook to Production](#from-notebook-to-production)
2. [Model Serialization](#model-serialization)
3. [API Deployment](#api-deployment)
4. [Containerization](#containerization)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Model Versioning](#model-versioning)
7. [Performance Optimization](#performance-optimization)
8. [Handling Model Drift](#handling-model-drift)

## From Notebook to Production

### The Gap Between Development and Production

**Development (Jupyter notebook):**
- Interactive experimentation
- Small datasets in memory
- Manual execution
- No reliability requirements
- Single user

**Production:**
- Automated pipelines
- Large-scale data
- Continuous operation
- High availability (99.9%+)
- Multiple concurrent users

### Production Requirements

A production ML system must:
- **Serve predictions reliably** - handle failures gracefully
- **Scale** - support expected load (requests per second)
- **Monitor performance** - track metrics in real-time
- **Update seamlessly** - deploy new models without downtime
- **Secure** - protect data and models
- **Audit** - log all predictions for debugging

### Production Readiness Checklist

- [ ] Model performance validated on test set
- [ ] Data preprocessing pipeline automated
- [ ] Model serialized and versioned
- [ ] API endpoint created and tested
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Monitoring dashboard set up
- [ ] Load testing completed
- [ ] Rollback plan documented
- [ ] Security review passed

## Model Serialization

### What is Serialization?

**Serialization** - converting a trained model into a format that can be saved to disk and loaded later for predictions.

### Pickle (Python's built-in)

```python
import pickle

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_new)
```

**Pros:**
- Simple, built-in
- Works with most scikit-learn models

**Cons:**
- Python version dependent
- Security risks (can execute arbitrary code)
- Large file sizes

### Joblib (Better for large NumPy arrays)

```python
import joblib

# Save model
joblib.dump(model, 'model.joblib')

# Load model
model = joblib.load('model.joblib')
```

**Pros:**
- More efficient for large NumPy arrays
- Better compression
- Recommended by scikit-learn

**Cons:**
- Still Python-specific
- Version compatibility issues

### ONNX (Cross-platform)

```python
# Convert scikit-learn model to ONNX
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, n_features]))]
onx = convert_sklearn(model, initial_types=initial_type)

# Save ONNX model
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# Load and use with ONNX Runtime
import onnxruntime as rt
sess = rt.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
predictions = sess.run(None, {input_name: X_new.astype(np.float32)})[0]
```

**Pros:**
- Language agnostic
- Optimized for inference
- Supported by many frameworks

**Cons:**
- More complex setup
- Not all models supported

### Saving Preprocessing Pipeline

**Always save the entire pipeline, not just the model:**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Save entire pipeline
joblib.dump(pipeline, 'pipeline.joblib')

# Load and use
pipeline = joblib.load('pipeline.joblib')
predictions = pipeline.predict(X_new)  # Scaling applied automatically
```

**Why this matters:**
- Preprocessing must be identical in production
- Prevents train-serve skew
- Single source of truth

### Model Metadata

Save metadata alongside model:

```python
import json
from datetime import datetime

metadata = {
    'model_type': 'RandomForestClassifier',
    'training_date': datetime.now().isoformat(),
    'features': list(X_train.columns),
    'performance': {
        'train_accuracy': 0.95,
        'test_accuracy': 0.92,
        'test_f1': 0.90
    },
    'hyperparameters': {
        'n_estimators': 100,
        'max_depth': 10
    },
    'data_version': 'v2.3',
    'scikit_learn_version': '1.3.0'
}

# Save metadata
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

## API Deployment

### Flask API (Simple)

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model at startup
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)

        # Return result
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Test the API:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### FastAPI (Production-ready)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load('model.joblib')

# Define request schema
class PredictionRequest(BaseModel):
    features: list[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

# Define response schema
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    status: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Prepare features
        features = np.array(request.features).reshape(1, -1)

        # Validate input
        if features.shape[1] != model.n_features_in_:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {model.n_features_in_} features, got {features.shape[1]}"
            )

        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)

        return PredictionResponse(
            prediction=int(prediction[0]),
            probability=float(probability[0][1]),
            status="success"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/model/info")
async def model_info():
    return {
        "n_features": model.n_features_in_,
        "n_classes": len(model.classes_),
        "model_type": type(model).__name__
    }
```

**Run with uvicorn:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Benefits of FastAPI:**
- Automatic API documentation (OpenAPI/Swagger)
- Request/response validation
- Async support
- Better performance than Flask
- Type hints

### Batch Prediction Endpoint

```python
@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    """Handle multiple predictions in one request"""
    try:
        # Stack all features
        all_features = np.array([req.features for req in requests])

        # Batch prediction (more efficient)
        predictions = model.predict(all_features)
        probabilities = model.predict_proba(all_features)

        # Format results
        results = [
            {
                'prediction': int(pred),
                'probability': float(prob[1]),
                'status': 'success'
            }
            for pred, prob in zip(predictions, probabilities)
        ]

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Containerization

### Why Containers?

**Benefits:**
- **Reproducibility** - same environment everywhere
- **Isolation** - dependencies don't conflict
- **Portability** - runs anywhere (local, cloud)
- **Scalability** - easy to replicate

### Dockerfile for ML API

```dockerfile
# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model.joblib .
COPY main.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt:**

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
scikit-learn==1.3.0
numpy==1.24.3
pydantic==2.5.0
joblib==1.3.2
```

### Build and Run Docker Container

```bash
# Build image
docker build -t ml-api:v1 .

# Run container
docker run -p 8000:8000 ml-api:v1

# Test
curl http://localhost:8000/health
```

### Docker Compose (Multi-service)

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/model.joblib
    volumes:
      - ./models:/app/models
    restart: always

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  monitoring:
    image: prom/prometheus
    ports:
      - "9090:9090"
```

**Run with:**

```bash
docker-compose up -d
```

## Monitoring and Logging

### Why Monitor?

**Track:**
- **Request volume** - traffic patterns
- **Response time** - latency issues
- **Error rate** - failures
- **Prediction distribution** - data drift
- **Resource usage** - CPU, memory

### Application Logging

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(request: PredictionRequest):
    start_time = datetime.now()

    try:
        # Log incoming request
        logger.info(f"Prediction request received: {request.features}")

        # Make prediction
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)
        probability = model.predict_proba(features)

        # Calculate latency
        latency = (datetime.now() - start_time).total_seconds()

        # Log successful prediction
        logger.info(
            f"Prediction: {prediction[0]}, "
            f"Probability: {probability[0][1]:.3f}, "
            f"Latency: {latency:.3f}s"
        )

        return {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'status': 'success'
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

### Metrics Collection with Prometheus

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

# Define metrics
predictions_total = Counter('predictions_total', 'Total predictions made')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
prediction_errors = Counter('prediction_errors_total', 'Total prediction errors')
model_score = Gauge('model_score', 'Current model score')

@app.post("/predict")
async def predict(request: PredictionRequest):
    with prediction_latency.time():
        try:
            # Make prediction
            prediction = model.predict(features)
            predictions_total.inc()
            return result

        except Exception as e:
            prediction_errors.inc()
            raise

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

### Prediction Storage for Analysis

```python
import sqlite3
from datetime import datetime

def log_prediction(features, prediction, probability):
    """Store predictions for later analysis"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO predictions (timestamp, features, prediction, probability)
        VALUES (?, ?, ?, ?)
    ''', (datetime.now().isoformat(), str(features), prediction, probability))

    conn.commit()
    conn.close()

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    # Log to database
    log_prediction(request.features, prediction[0], probability[0][1])

    return result
```

## Model Versioning

### Why Version Models?

**Reasons:**
- Track which model version is deployed
- Rollback to previous version if needed
- A/B test different models
- Audit compliance

### Semantic Versioning

**Format:** `MAJOR.MINOR.PATCH`

- **MAJOR** - incompatible API changes (different features)
- **MINOR** - new functionality, backward compatible
- **PATCH** - bug fixes, same functionality

**Example:**
- `v1.0.0` - initial production model
- `v1.1.0` - improved feature engineering
- `v1.1.1` - bug fix in preprocessing
- `v2.0.0` - new features added (breaking change)

### Directory Structure

```
models/
├── v1.0.0/
│   ├── model.joblib
│   ├── metadata.json
│   └── requirements.txt
├── v1.1.0/
│   ├── model.joblib
│   ├── metadata.json
│   └── requirements.txt
└── production/  # Symlink to current production version
    └── -> v1.1.0/
```

### Loading Versioned Models

```python
import os

MODEL_DIR = os.getenv('MODEL_DIR', 'models/production')

def load_model(version='production'):
    """Load specific model version"""
    model_path = f'models/{version}/model.joblib'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model version {version} not found")

    model = joblib.load(model_path)

    # Load metadata
    with open(f'models/{version}/metadata.json', 'r') as f:
        metadata = json.load(f)

    return model, metadata

# Load model at startup
model, metadata = load_model()

@app.get("/model/version")
async def get_model_version():
    return metadata
```

### Model Registry (MLflow)

```python
import mlflow
from mlflow.tracking import MlflowClient

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Register model
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="my_classifier"
)

# Promote to production
client.transition_model_version_stage(
    name="my_classifier",
    version=3,
    stage="Production"
)

# Load production model
model = mlflow.pyfunc.load_model(
    model_uri="models:/my_classifier/Production"
)
```

## Performance Optimization

### Reduce Latency

**1. Use faster serialization formats:**

```python
# ONNX for faster inference
import onnxruntime as rt

sess = rt.InferenceSession("model.onnx")
predictions = sess.run(None, {input_name: X})[0]
```

**2. Batch predictions:**

```python
# Process multiple predictions together
predictions = model.predict(np.vstack([req1, req2, req3]))
```

**3. Cache predictions:**

```python
import redis
import hashlib
import json

redis_client = redis.Redis(host='localhost', port=6379)

def get_cache_key(features):
    """Generate cache key from features"""
    return hashlib.md5(str(features).encode()).hexdigest()

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Check cache
    cache_key = get_cache_key(request.features)
    cached = redis_client.get(cache_key)

    if cached:
        return json.loads(cached)

    # Make prediction
    prediction = model.predict(features)

    # Cache result (expire after 1 hour)
    result = {'prediction': int(prediction[0])}
    redis_client.setex(cache_key, 3600, json.dumps(result))

    return result
```

**4. Use faster models:**

```python
# Trade accuracy for speed
# Instead of: XGBoost with 1000 trees
slow_model = XGBClassifier(n_estimators=1000)

# Use: Fewer trees or simpler model
fast_model = XGBClassifier(n_estimators=100)  # 10x faster
# Or: LogisticRegression()  # 100x faster
```

### Scale Horizontally

**Load balancing with multiple workers:**

```bash
# Run multiple Uvicorn workers
uvicorn main:app --workers 8 --host 0.0.0.0 --port 8000
```

**Kubernetes deployment:**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 5  # Run 5 instances
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: api
        image: ml-api:v1
        ports:
        - containerPort: 8000
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### Optimize Resource Usage

```python
# Limit number of threads
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# For XGBoost
model = XGBClassifier(n_jobs=1)  # Limit CPU usage
```

## Handling Model Drift

### What is Model Drift?

**Concept drift** - statistical properties of target variable change over time.

**Data drift** - distribution of input features changes over time.

**Example:**
- E-commerce model trained pre-COVID performs poorly post-COVID
- Fraud detection model becomes less effective as fraudsters adapt

### Detecting Drift

**Monitor prediction distribution:**

```python
import numpy as np
from scipy import stats

# Store predictions over time
recent_predictions = []  # Last week
historical_predictions = []  # Training period

# Statistical test for distribution change
statistic, p_value = stats.ks_2samp(recent_predictions, historical_predictions)

if p_value < 0.05:
    logger.warning("Significant drift detected in predictions!")
    # Trigger retraining or alert
```

**Monitor feature distributions:**

```python
def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index (PSI)"""

    def scale_range(arr, buckets):
        """Bin continuous values"""
        return np.histogram(arr, bins=buckets)[0]

    expected_percents = scale_range(expected, buckets) / len(expected)
    actual_percents = scale_range(actual, buckets) / len(actual)

    # PSI calculation
    psi_values = (expected_percents - actual_percents) * \
                 np.log(expected_percents / (actual_percents + 1e-10))

    psi = np.sum(psi_values)

    return psi

# Calculate PSI for each feature
for feature in features:
    psi = calculate_psi(train_data[feature], production_data[feature])

    if psi > 0.2:
        logger.warning(f"High drift detected in feature {feature}: PSI={psi:.3f}")
```

**PSI interpretation:**
- PSI < 0.1: No significant change
- 0.1 < PSI < 0.2: Moderate change
- PSI > 0.2: Significant change (action needed)

### Performance Monitoring

```python
# Track model performance over time
def monitor_model_performance():
    """Calculate rolling metrics on recent predictions"""

    # Get recent predictions and actuals (when labels become available)
    recent_data = get_recent_labeled_data()

    if len(recent_data) > 100:
        y_true = recent_data['actual']
        y_pred = recent_data['predicted']

        # Calculate metrics
        current_accuracy = accuracy_score(y_true, y_pred)
        current_f1 = f1_score(y_true, y_pred)

        # Compare to training performance
        if current_accuracy < training_accuracy - 0.05:
            logger.warning(
                f"Model accuracy dropped: {current_accuracy:.3f} "
                f"(training: {training_accuracy:.3f})"
            )
            # Trigger retraining
```

### Retraining Strategies

**1. Scheduled retraining:**
```python
# Retrain every week
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(retrain_model, 'cron', day_of_week='mon', hour=2)
scheduler.start()

def retrain_model():
    """Retrain model with latest data"""
    logger.info("Starting scheduled retraining...")

    # Fetch latest data
    new_data = fetch_new_training_data()

    # Train new model
    new_model = train_pipeline(new_data)

    # Validate performance
    if validate_model(new_model):
        # Save and deploy
        deploy_model(new_model, version='v1.2.0')
        logger.info("Model retrained and deployed successfully")
```

**2. Trigger-based retraining:**
```python
# Retrain when drift detected
if drift_detected or performance_degraded:
    retrain_model()
```

**3. Online learning:**
```python
# Incrementally update model with new data
@app.post("/feedback")
async def update_model(features, true_label):
    """Update model with new labeled example"""
    model.partial_fit([features], [true_label])
    logger.info("Model updated with new example")
```

## Deployment Checklist

Before deploying to production:

**Model:**
- [ ] Model performance acceptable on test set
- [ ] Model tested on edge cases
- [ ] Model serialized correctly
- [ ] Preprocessing pipeline included

**API:**
- [ ] Input validation implemented
- [ ] Error handling comprehensive
- [ ] Health check endpoint working
- [ ] API documentation complete
- [ ] Rate limiting configured

**Infrastructure:**
- [ ] Containerized and tested
- [ ] Environment variables configured
- [ ] Secrets management set up
- [ ] Load balancing configured
- [ ] Auto-scaling rules defined

**Monitoring:**
- [ ] Logging configured
- [ ] Metrics collection working
- [ ] Alerts set up
- [ ] Dashboards created
- [ ] Drift detection enabled

**Operations:**
- [ ] Deployment procedure documented
- [ ] Rollback plan tested
- [ ] On-call rotation established
- [ ] Runbook created

## Example Production Architecture

```
                     ┌─────────────┐
                     │   Client    │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │Load Balancer│
                     └──────┬──────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
      ┌────▼────┐      ┌────▼────┐     ┌────▼────┐
      │ API (1) │      │ API (2) │     │ API (3) │
      └────┬────┘      └────┬────┘     └────┬────┘
           │                │                │
           └────────────────┼────────────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
      ┌────▼────┐      ┌────▼────┐     ┌────▼─────┐
      │  Redis  │      │Database │     │Monitoring│
      │ (Cache) │      │  (Logs) │     │(Prometheus)│
      └─────────┘      └─────────┘     └──────────┘
```

## References

- "Designing Machine Learning Systems" (Chip Huyen)
- "Building Machine Learning Powered Applications" (Emmanuel Ameisen)
- "Machine Learning Engineering" (Andriy Burkov)
- FastAPI Documentation: https://fastapi.tiangolo.com/
- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- Docker Documentation: https://docs.docker.com/
