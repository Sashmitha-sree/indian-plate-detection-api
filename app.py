from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io, os

app = FastAPI(title="License Plate Detection API")

model = None

@app.on_event("startup")
def load_model():
    global model
    from ultralytics import RTDETR
    if os.path.exists("best.pt"):
        model = RTDETR("best.pt")

# ── PART A: Detection ────────────────────────────────────────
@app.post("/detect")
async def detect_plates(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = model.predict(image, conf=0.3)
    
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": "license-plate",
                "confidence": round(float(box.conf), 3),
                "bbox": {
                    "x1": round(float(box.xyxy[0][0]), 1),
                    "y1": round(float(box.xyxy[0][1]), 1),
                    "x2": round(float(box.xyxy[0][2]), 1),
                    "y2": round(float(box.xyxy[0][3]), 1)
                }
            })
    
    return JSONResponse({
        "status": "success",
        "filename": file.filename,
        "detections": detections,
        "total_plates_found": len(detections)
    })

# ── PART B: Reasoning ────────────────────────────────────────
@app.post("/reason")
async def reason_about_image(
    question: str,
    file: UploadFile = File(...)
):
    question_lower = question.lower()

    # Hand-written intent router — no frameworks
    IMAGE_KEYWORDS = [
        "plate", "number", "license", "vehicle", "car",
        "how many", "detect", "found", "registration",
        "visible", "count", "identify"
    ]
    needs_detection = any(kw in question_lower for kw in IMAGE_KEYWORDS)

    if not needs_detection:
        return JSONResponse({
            "question": question,
            "answer": "This question does not appear to be about license plates or vehicles in the image. Please ask something related to the image content.",
            "detection_used": False
        })

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = model.predict(image, conf=0.3)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": "license-plate",
                "confidence": round(float(box.conf), 3),
                "bbox": {
                    "x1": round(float(box.xyxy[0][0]), 1),
                    "y1": round(float(box.xyxy[0][1]), 1),
                    "x2": round(float(box.xyxy[0][2]), 1),
                    "y2": round(float(box.xyxy[0][3]), 1)
                }
            })

    # Confidence guardrail
    if len(detections) == 0:
        return JSONResponse({
            "question": question,
            "answer": "Insufficient information: No license plates were detected with sufficient confidence. The image may be unclear, too dark, or no plates are visible.",
            "detection_used": True,
            "detections": []
        })

    avg_conf = sum(d["confidence"] for d in detections) / len(detections)
    if avg_conf < 0.4:
        return JSONResponse({
            "question": question,
            "answer": f"Insufficient information: Detection confidence too low (avg: {avg_conf:.2f}). Cannot answer reliably.",
            "detection_used": True,
            "detections": detections
        })

    # Reasoning over structured output
    count = len(detections)
    high_conf = [d for d in detections if d["confidence"] > 0.7]

    if "how many" in question_lower or "count" in question_lower:
        answer = f"{count} license plate(s) detected with average confidence of {avg_conf:.2f}."
    elif "visible" in question_lower or "detect" in question_lower:
        answer = f"Yes, {count} license plate(s) are visible. Confidence scores: {[d['confidence'] for d in detections]}."
    else:
        answer = f"Found {count} license plate(s). {len(high_conf)} detected with high confidence (>0.7)."

    return JSONResponse({
        "question": question,
        "answer": answer,
        "detection_used": True,
        "detections": detections,
        "avg_confidence": round(avg_conf, 3)
    })

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
