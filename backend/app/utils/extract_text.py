import fitz


def extract_text(pdf_path):
    """
    Extract PDF text while preserving page boundaries.
    """

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text().strip()

        if text:
            pages.append({
                "text": text,
                "source": pdf_path.split("\\")[-1],
                "page": page_number
            })

    document.close()

    return pages