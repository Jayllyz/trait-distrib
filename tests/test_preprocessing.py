from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image

from postal_app.preprocessing import (
    ImageValidationError,
    SegmentationError,
    load_image,
    segment_postal_code,
    stack_segment_images,
)


def _postal_code_image(code: str = "75015") -> np.ndarray:
    image = np.full((260, 920, 3), 255, dtype=np.uint8)
    for index, digit in enumerate(code):
        cv2.putText(
            image,
            digit,
            (55 + index * 175, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            4.8,
            (0, 0, 0),
            13,
            cv2.LINE_AA,
        )
    return image


def test_load_image_decodes_png() -> None:
    buffer = BytesIO()
    Image.fromarray(_postal_code_image()).save(buffer, format="PNG")

    loaded = load_image(buffer.getvalue())

    assert loaded.shape == (260, 920, 3)
    assert loaded.dtype == np.uint8


def test_load_image_rejects_empty_or_tiny_files() -> None:
    with pytest.raises(ImageValidationError, match="vide"):
        load_image(b"")

    buffer = BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, format="PNG")
    with pytest.raises(ImageValidationError, match="trop petite"):
        load_image(buffer.getvalue())


def test_segment_five_left_to_right_digits() -> None:
    segments = segment_postal_code(_postal_code_image())
    batch = stack_segment_images(segments)

    assert len(segments) == 5
    assert batch.shape == (5, 28, 28)
    assert batch.dtype == np.uint8
    assert all(np.count_nonzero(digit) > 0 for digit in batch)
    assert [segment.bounding_box[0] for segment in segments] == sorted(
        segment.bounding_box[0] for segment in segments
    )


def test_small_noise_is_ignored() -> None:
    image = _postal_code_image()
    image[15, 15] = 0
    image[240, 900] = 0

    assert len(segment_postal_code(image)) == 5


def test_uneven_phone_lighting_does_not_merge_all_digits() -> None:
    background = np.linspace(75, 170, 920, dtype=np.uint8)
    image = np.repeat(background[np.newaxis, :], 260, axis=0)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    for index, digit in enumerate("93400"):
        cv2.putText(
            image,
            digit,
            (55 + index * 175, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            4.8,
            (35, 35, 35),
            10,
            cv2.LINE_AA,
        )

    assert len(segment_postal_code(image)) == 5


def test_blank_image_is_rejected() -> None:
    with pytest.raises(SegmentationError, match="Aucune écriture"):
        segment_postal_code(np.full((200, 500, 3), 255, dtype=np.uint8))


def test_wrong_number_of_regions_is_rejected() -> None:
    image = np.full((220, 600, 3), 255, dtype=np.uint8)
    for x in (60, 250, 440):
        cv2.line(image, (x, 40), (x, 185), (0, 0, 0), 5)

    with pytest.raises(SegmentationError, match="au lieu de 5"):
        segment_postal_code(image)
