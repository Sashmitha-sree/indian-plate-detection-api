# Indian License Plate Detection API

RT-DETR based license plate detection exposed via FastAPI.

## Run locally
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

## Endpoints

### POST /detect
Detects license plates in an image.
curl -X POST "http://localhost:8000/detect" -F "file=@car.jpg"

### POST /reason
Answer natural language questions about plates in image.
curl -X POST "http://localhost:8000/reason" \
  -F "question=How many plates are visible?" \
  -F "file=@car.jpg"

### GET /health
curl http://localhost:8000/health
