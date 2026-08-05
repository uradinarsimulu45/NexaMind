from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.utils.extract_text import extract_text
from app.utils.extract_images import extract_images
from app.utils.text_chunker import chunk_text

router = APIRouter()

UPLOAD_FOLDER = "data/pdfs"
IMAGE_FOLDER = "data/images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text
    text = extract_text(file_path)

    # Extract images
    image_count = extract_images(file_path, IMAGE_FOLDER)

    # Chunk text
    chunks = chunk_text(text)

    return {
        "message": "PDF uploaded successfully",
        "filename": file.filename,
        "characters": len(text),
        "images": image_count,
        "chunks": len(chunks),
        "first_chunk": chunks[0] if chunks else ""
    }