from fastapi import APIRouter, UploadFile, File
from app.utils.pdf_extractor import extract_text
import os
import shutil

router = APIRouter()

UPLOAD_DIR = "../data/pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(file_path)

    return {
        "message": "PDF uploaded successfully",
        "filename": file.filename,
        "characters": len(text),
        "preview": text[:500]
    }