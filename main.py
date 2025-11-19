from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uuid
import shutil
import cv2
from ultralytics import YOLO
import os

app = FastAPI(title="PPE Detection FastAPI")

# HTML templates
templates = Jinja2Templates(directory="templates")

# Load YOLO model once
model = YOLO("best.pt")

@app.get("/", response_class=HTMLResponse)
async def frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_video(file: UploadFile = File(...)):
    # Save uploaded file
    uid = uuid.uuid4().hex
    input_path = f"input_{uid}.mp4"
    output_path = f"output_{uid}.mp4"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {"error": "Cannot open video"}

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame, conf=0.5)

        # Draw results
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", 
                            (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                            (0,255,0), 2)

        writer.write(frame)

    cap.release()
    writer.release()

    return FileResponse(output_path,
                        media_type="video/mp4",
                        filename="processed.mp4")
