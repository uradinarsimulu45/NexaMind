from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.utils.extract_text import extract_text
from app.utils.extract_images import extract_images
from app.utils.text_chunker import chunk_text

from app.vector_db.embeddings import generate_embeddings
from app.vector_db.faiss_store import create_faiss_index

router = APIRouter()

UPLOAD_FOLDER = "data/pdfs"
IMAGE_FOLDER = "data/images"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Save uploaded PDF
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text
    text = extract_text(file_path)

    # Extract images
    image_count = extract_images(file_path, IMAGE_FOLDER)

    # Split text into chunks
    chunks = chunk_text(text)

    # Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Store embeddings in FAISS
    stored_vectors = create_faiss_index(embeddings)

    # API Response
    return {
        "message": "PDF uploaded successfully",
        "filename": file.filename,
        "characters": len(text),
        "images": image_count,
        "chunks": len(chunks),
        "stored_vectors": stored_vectors,
        "first_chunk": chunks[0] if chunks else ""
    }