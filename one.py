from fastapi import FastAPI, File, UploadFile
import cv2
import os
from tempfile import NamedTemporaryFile
import requests

app = FastAPI()

harcascade = "model/haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(harcascade)

process_dir = "process"
os.makedirs(process_dir, exist_ok=True)

# node_server_url = "http://localhost:3000/upload"
node_server_url = "https://lynx-fun-normally.ngrok-free.app/upload"

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_file_path = temp_file.name

    img = cv2.imread(temp_file_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4)
    min_area = 500
    scanned_images = []

    for count, (x, y, w, h) in enumerate(plates):
        area = w * h
        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            scanned_img_path = os.path.join(process_dir, f"scanned_img_{count}.jpg")
            img_roi = img[y: y + h, x: x + w]
            cv2.imwrite(scanned_img_path, img_roi)
            scanned_images.append(scanned_img_path)

    os.unlink(temp_file_path)

    for scanned_img in scanned_images:
        with open(scanned_img, 'rb') as img_file:
            files = {'file': img_file}
            response = requests.post(node_server_url, files=files)

    return {"scanned_images": scanned_images}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
