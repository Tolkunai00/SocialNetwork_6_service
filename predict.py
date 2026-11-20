import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO

app = FastAPI(title="YOLO Object Detection API")

model = YOLO("yolov8n.pt")


@app.post("/check_image/")
async def check_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    results = model(image)

    objects = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            objects.append({"class": class_name})

    return {"objects": objects}


@app.post("/check_video/")
async def check_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    data = await file.read()

    temp_file = "temp_video.mp4"

    with open(temp_file, "wb") as f:
        f.write(data)

    video = cv2.VideoCapture(temp_file)

    if not video.isOpened():
        raise HTTPException(status_code=400, detail="Invalid video")

    objects = []
    seen_classes = set()

    while True:
        ret, frame = video.read()

        if not ret:
            break

        results = model(frame)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if class_name not in seen_classes:
                    seen_classes.add(class_name)
                    objects.append({"class": class_name})

    video.release()

    return {"objects": objects}