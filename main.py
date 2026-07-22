import cv2
import asyncio
import PIL.Image
import io
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import base64
import os

app = FastAPI(title="Camera Defect Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## Load The Model
model = YOLO("models/best.pt")


# ***********************************************************************************************************
## Camera Class (Safe for Cloud Servers)
class GlobalCamera:
    def __init__(self):
        self.camera = None

    def get_instance(self):
        if self.camera is None or not self.camera.isOpened():
            # تحسين فتح الكاميرا وتجنب الـ Crash إذا لم تكن موجودة
            try:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    return None
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            except Exception as e:
                print(f"⚠️ Camera initialisation error: {e}")
                return None
        return self.camera

    def release(self):
        if self.camera and self.camera.isOpened():
            self.camera.release()
            self.camera = None
            print("🔴 Camera resources released successfully.")


cam_obj = GlobalCamera()


# ****************************************************************************************************
@app.get("/")
def root():
    return {
        "status": "API is running",
        "Supervised": "Al-Azher Camera Defect Detection Team",
    }


# ******************************************************************************************************
## Analysis Image endpoint
@app.post("/predict")
async def predict_defect(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = PIL.Image.open(io.BytesIO(image_bytes))
        results = model(image, conf=0.25)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append(
                    {
                        "class": model.names[int(box.cls[0])],
                        "confidence": round(float(box.conf[0]), 4),
                        "box": [round(x, 2) for x in box.xyxy[0].tolist()],
                    }
                )
        return {
            "status": "success",
            "total_defects": len(detections),
            "detections": detections,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# *****************************************************************************************************
## Live Video endpoint (Only active if local physical camera exists)
async def generate_live_frames():
    camera = cam_obj.get_instance()

    if camera is None or not camera.isOpened():
        print("Error: No hardware camera available on server environment.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            await asyncio.sleep(0.03)
            continue

        results = model(frame, stream=True, conf=0.5, verbose=False)

        annotated_frame = frame
        for r in results:
            annotated_frame = r.plot()

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

        await asyncio.sleep(0.03)  # FPS => 30


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_live_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ****************************************************************************************************
## Server Shutdown Cleanup
@app.on_event("shutdown")
def shutdown_event():
    cam_obj.release()
    cv2.destroyAllWindows()
    print("Server shutting down cleanly.")


# *****************************************************************************************************
## WebSocket Endpoint (Primary Stream Solution for Remote Deployments)
@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("⚡ Client connected via WebSocket")
    try:
        while True:
            data = await websocket.receive_text()
            
            if not data or "," not in data:
                continue

            ## open Hashing
            image_data = base64.b64decode(data.split(",")[1])
            image = PIL.Image.open(io.BytesIO(image_data))

            ## Inference
            results = model(image, conf=0.5, verbose=False)

            detections = []
            for r in results:
                for box in r.boxes:
                    detections.append(
                        {
                            "class": model.names[int(box.cls[0])],
                            "confidence": round(float(box.conf[0]), 4),
                            "box": [round(x, 2) for x in box.xyxy[0].tolist()],
                        }
                    )

            ## Send to the front
            await websocket.send_json(
                {
                    "status": "success",
                    "total_defects": len(detections),
                    "detections": detections,
                }
            )

    except WebSocketDisconnect:
        print("🔴 Client disconnected from WebSocket")
    except Exception as e:
        print(f"WebSocket Error: {e}")
