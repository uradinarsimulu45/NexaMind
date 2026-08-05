import fitz
import os

def extract_images(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    pdf = fitz.open(pdf_path)
    image_count = 0

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        images = page.get_images(full=True)

        for img in images:
            xref = img[0]
            base_image = pdf.extract_image(xref)

            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_path = os.path.join(
                output_folder,
                f"page_{page_num+1}_img_{image_count}.{image_ext}"
            )

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_count += 1

    pdf.close()
    return image_count