from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.utils.extract_text import extract_text
from app.utils.extract_images import extract_images

from app.vector_db.embeddings import generate_embeddings
from app.vector_db.faiss_store import create_faiss_index


router = APIRouter()


UPLOAD_FOLDER = "data/pdfs"
IMAGE_FOLDER = "data/images"


# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # --------------------------------
    # 1. Save uploaded PDF
    # --------------------------------

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )


    # --------------------------------
    # 2. Extract text page-by-page
    # --------------------------------

    pages = extract_text(file_path)


    # --------------------------------
    # 3. Extract images
    # --------------------------------

    image_count = extract_images(
        file_path,
        IMAGE_FOLDER
    )


    # --------------------------------
    # 4. Use each page as a document
    # --------------------------------

    chunks = pages


    # --------------------------------
    # 5. Get text from each page
    # --------------------------------

    texts = [
        page["text"]
        for page in chunks
    ]


    # --------------------------------
    # 6. Generate embeddings
    # --------------------------------

    embeddings = generate_embeddings(
        texts
    )


    # --------------------------------
    # 7. Store embeddings in FAISS
    # --------------------------------

    stored_vectors = create_faiss_index(
        embeddings,
        chunks
    )


    # --------------------------------
    # 8. Return response
    # --------------------------------

    return {
        "message": "PDF uploaded successfully",
        "filename": file.filename,
        "pages": len(pages),
        "images": image_count,
        "chunks": len(chunks),
        "stored_vectors": stored_vectors,
        "first_chunk": (
            chunks[0]
            if chunks
            else {}
        )
    }