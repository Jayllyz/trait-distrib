from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_PROCESSING_SIDE = 2_000
EXPECTED_DIGITS = 5


class ImageValidationError(ValueError):
    pass


class SegmentationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DigitSegment:
    image: np.ndarray
    bounding_box: tuple[int, int, int, int]


def load_image(data: bytes) -> np.ndarray:
    if not data:
        raise ImageValidationError("L'image reçue est vide.")
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise ImageValidationError("L'image dépasse la taille maximale de 10 Mo.")

    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ImageValidationError(
                    "Seuls les fichiers JPG et PNG sont acceptés."
                )
            width, height = source.size
            if width * height > MAX_IMAGE_PIXELS:
                raise ImageValidationError("La résolution de l'image est trop élevée.")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageValidationError(
            "Le fichier ne contient pas une image valide."
        ) from error

    if min(image.size) < 80:
        raise ImageValidationError(
            "L'image est trop petite pour distinguer cinq chiffres."
        )

    image.thumbnail(
        (MAX_PROCESSING_SIDE, MAX_PROCESSING_SIDE), Image.Resampling.LANCZOS
    )
    return np.asarray(image)


def segment_postal_code(image_rgb: np.ndarray) -> tuple[DigitSegment, ...]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ImageValidationError("L'image doit être fournie au format RGB.")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # A phone photo rarely has a perfectly uniform white background. A global
    # Otsu threshold can therefore turn a shadow covering half the sheet into
    # one giant foreground region. Compare each pixel with its local
    # neighbourhood instead, so that only dark pen strokes remain foreground.
    shortest_side = min(gray.shape)
    block_size = min(101, max(31, shortest_side // 8))
    if block_size % 2 == 0:
        block_size += 1
    foreground = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        7,
    )
    foreground = _remove_small_components(foreground)

    points = cv2.findNonZero(foreground)
    if points is None:
        raise SegmentationError(
            "Aucune écriture n'a été détectée. Utilisez un fond uni et un stylo foncé."
        )

    x, y, width, height = cv2.boundingRect(points)
    if width < EXPECTED_DIGITS * 3 or height < 8:
        raise SegmentationError("La zone écrite détectée est trop petite.")

    horizontal_margin = max(2, width // 100)
    vertical_margin = max(2, height // 12)
    x0 = max(0, x - horizontal_margin)
    y0 = max(0, y - vertical_margin)
    x1 = min(foreground.shape[1], x + width + horizontal_margin)
    y1 = min(foreground.shape[0], y + height + vertical_margin)
    cropped = foreground[y0:y1, x0:x1]

    runs = _find_horizontal_runs(cropped)
    if len(runs) > EXPECTED_DIGITS:
        runs = _merge_fragmented_runs(runs, EXPECTED_DIGITS)
    if len(runs) < EXPECTED_DIGITS:
        runs = _split_wide_runs(cropped, runs, EXPECTED_DIGITS)

    if len(runs) != EXPECTED_DIGITS:
        raise SegmentationError(
            f"{len(runs)} zone(s) ont été détectées au lieu de 5. "
            "Espacez davantage les chiffres et reprenez la photo."
        )

    segments: list[DigitSegment] = []
    for start, end in sorted(runs):
        digit_mask = cropped[:, start:end]
        digit_points = cv2.findNonZero(digit_mask)
        if digit_points is None:
            raise SegmentationError("Une zone détectée ne contient aucun chiffre.")
        digit_x, digit_y, digit_width, digit_height = cv2.boundingRect(digit_points)
        if digit_height < max(5, int(height * 0.35)):
            raise SegmentationError(
                "Une zone ressemble à une tache plutôt qu'à un chiffre. Reprenez la photo."
            )
        isolated = digit_mask[
            digit_y : digit_y + digit_height, digit_x : digit_x + digit_width
        ]
        normalized = _normalize_mnist(isolated)
        segments.append(
            DigitSegment(
                image=normalized,
                bounding_box=(
                    x0 + start + digit_x,
                    y0 + digit_y,
                    digit_width,
                    digit_height,
                ),
            )
        )

    return tuple(segments)


def thicken_segments(
    segments: tuple[DigitSegment, ...],
) -> tuple[DigitSegment, ...]:
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return tuple(
        DigitSegment(
            image=cv2.dilate(segment.image, kernel, iterations=1),
            bounding_box=segment.bounding_box,
        )
        for segment in segments
    )


def stack_segment_images(segments: tuple[DigitSegment, ...]) -> np.ndarray:
    if len(segments) != EXPECTED_DIGITS:
        raise ValueError("exactly five segments are required")
    return np.stack([segment.image for segment in segments]).astype(np.uint8)


def _remove_small_components(binary: np.ndarray) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    image_area = binary.shape[0] * binary.shape[1]
    minimum_area = max(12, int(image_area * 0.00002))
    minimum_height = max(4, int(binary.shape[0] * 0.012))
    cleaned = np.zeros_like(binary)

    for label in range(1, component_count):
        _, _, width, height, area = stats[label]
        if area >= minimum_area and height >= minimum_height and width >= 1:
            cleaned[labels == label] = 255
    return cleaned


def _find_horizontal_runs(binary: np.ndarray) -> list[tuple[int, int]]:
    occupied = np.any(binary > 0, axis=0).astype(np.uint8)
    if not np.any(occupied):
        return []

    # Close tiny internal gaps caused by disconnected pen strokes while keeping
    # the visible spaces between separately written digits.
    close_width = max(2, binary.shape[1] // 250)
    kernel = np.ones(close_width, dtype=np.uint8)
    occupied = np.convolve(occupied, kernel, mode="same") > 0
    changes = np.diff(np.pad(occupied.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if end - start >= 2
    ]


def _merge_fragmented_runs(
    runs: list[tuple[int, int]], expected: int
) -> list[tuple[int, int]]:
    merged = list(runs)
    while len(merged) > expected:
        gaps = [
            merged[index + 1][0] - merged[index][1] for index in range(len(merged) - 1)
        ]
        merge_at = int(np.argmin(gaps))
        combined = (merged[merge_at][0], merged[merge_at + 1][1])
        merged[merge_at : merge_at + 2] = [combined]
    return merged


def _split_wide_runs(
    binary: np.ndarray, runs: list[tuple[int, int]], expected: int
) -> list[tuple[int, int]]:
    split_runs = list(runs)
    while split_runs and len(split_runs) < expected:
        widths = [end - start for start, end in split_runs]
        widest_index = int(np.argmax(widths))
        start, end = split_runs[widest_index]
        width = end - start
        if width < 8:
            break

        projection = np.count_nonzero(binary[:, start:end], axis=0)
        lower = max(3, int(width * 0.25))
        upper = min(width - 3, int(width * 0.75))
        if lower >= upper:
            break
        valley_offset = lower + int(np.argmin(projection[lower:upper]))
        peak = int(np.max(projection))
        if peak == 0 or projection[valley_offset] > peak * 0.35:
            break
        cut = start + valley_offset
        split_runs[widest_index : widest_index + 1] = [(start, cut), (cut, end)]
    return split_runs


def _normalize_mnist(digit: np.ndarray) -> np.ndarray:
    height, width = digit.shape
    scale = min(20 / max(width, 1), 20 / max(height, 1))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(
        digit, (target_width, target_height), interpolation=interpolation
    )
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x = (28 - target_width) // 2
    y = (28 - target_height) // 2
    canvas[y : y + target_height, x : x + target_width] = resized

    moments = cv2.moments(canvas)
    if moments["m00"]:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        transform = np.asarray(
            [[1, 0, 13.5 - center_x], [0, 1, 13.5 - center_y]],
            dtype=np.float32,
        )
        canvas = cv2.warpAffine(canvas, transform, (28, 28), borderValue=0)
    return canvas
