from fastapi import APIRouter, UploadFile, File
from app.utils.pdf_extractor import extract_text
from app.utils.image_extractor import extract_images
import os
import shutil

router = APIRouter()

UPLOAD_DIR = "../data/pdfs"
IMAGE_DIR = "../data/images"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Save uploaded PDF
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text
    text = extract_text(file_path)

    # Extract images
    image_count = extract_images(file_path, IMAGE_DIR)

    # Return response
    return {
        "message": "PDF uploaded successfully",
        "filename": file.filename,
        "characters": len(text),
        "images": image_count,
        "preview": text[:500]
    }