import os
import glob


IMAGE_FOLDER = "data/images"


def get_available_images():
    """
    Return all extracted images from the PDF.
    """

    extensions = ["*.jpg", "*.jpeg", "*.png"]

    images = []

    for extension in extensions:
        images.extend(
            glob.glob(
                os.path.join(IMAGE_FOLDER, extension)
            )
        )

    return sorted(images)


def select_image(question: str):
    """
    Select an image based on simple question keywords.

    Day 17:
    Uses filename/page information as a lightweight
    image-selection mechanism.
    """

    images = get_available_images()

    if not images:
        return None

    question = question.lower()

    # NASA lunar / Mars / spacecraft questions
    if any(word in question for word in [
        "moon",
        "lunar",
        "mars",
        "planet",
        "spacecraft",
        "landing",
        "gateway"
    ]):

        # Prefer images from page 14 for the current NASA document.
        page_14_images = [
            image for image in images
            if "page_14_" in image.lower()
        ]

        if page_14_images:
            return page_14_images[0]

    # Otherwise use the first available image.
    return images[0]