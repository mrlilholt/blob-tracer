# backend/api.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import tempfile

from blob_detection import detect_blobs_in_video

app = FastAPI()

# Allow your Netlify site to call this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, restrict to your Netlify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Where to store temp files (you can adjust if your host requires)
TMP_DIR = tempfile.gettempdir()


@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    max_boxes: int = Form(None),
    min_area: int = Form(50),
):
    # Create unique filenames
    file_id = str(uuid.uuid4())
    input_path = os.path.join(TMP_DIR, f"{file_id}_{file.filename}")
    output_path = os.path.join(TMP_DIR, f"{file_id}_output.mp4")

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Run your existing blob detection on the video
    detect_blobs_in_video(
        video_path=input_path,
        output_path=output_path,
        max_boxes=max_boxes,
        min_area=min_area,
    )

    # Return processed video
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="blob_traced_output.mp4"
    )
