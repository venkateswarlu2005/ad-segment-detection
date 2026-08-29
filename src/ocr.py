from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re

import pytesseract
from PIL import Image


@dataclass
class OCRWord:
    text: str
    confidence: float


@dataclass
class OCRResult:
    timestamp: float
    frame: str
    raw_text: str
    clean_text: str
    confidence: float
    words: list[OCRWord]
    usable: bool


class OCRProcessor:
    """
    OCR processor for extracting visible text from video frames.

    Keeps both:
        raw_text   -> original Tesseract output
        clean_text -> filtered OCR output

    This is intentionally NOT an advertisement detector.
    """

    def __init__(
        self,
        output_dir: str | Path = "data/outputs",
        minimum_confidence: float = 30.0,
        minimum_word_confidence: float = 30.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.minimum_confidence = minimum_confidence
        self.minimum_word_confidence = (
            minimum_word_confidence
        )

        print("Tesseract OCR initialized.")
        print(
            f"Minimum frame confidence: "
            f"{self.minimum_confidence}"
        )
        print(
            f"Minimum word confidence: "
            f"{self.minimum_word_confidence}"
        )

    # =========================================================
    # SINGLE FRAME
    # =========================================================

    def process_frame(
        self,
        frame_path: str | Path,
        timestamp: float,
    ) -> OCRResult:

        frame_path = Path(frame_path)

        if not frame_path.exists():
            raise FileNotFoundError(
                f"Frame not found: {frame_path}"
            )

        with Image.open(frame_path) as image:
            image = image.convert("RGB")

            data = pytesseract.image_to_data(
                image,
                config="--psm 6",
                output_type=pytesseract.Output.DICT,
            )

        words: list[OCRWord] = []

        for index, raw_text in enumerate(
            data["text"]
        ):
            text = raw_text.strip()

            if not text:
                continue

            try:
                confidence = float(
                    data["conf"][index]
                )
            except (
                ValueError,
                TypeError,
            ):
                continue

            if confidence < 0:
                continue

            words.append(
                OCRWord(
                    text=text,
                    confidence=confidence,
                )
            )

        raw_text = " ".join(
            word.text
            for word in words
        ).strip()

        clean_words = self.clean_words(words)

        clean_text = " ".join(
            word.text
            for word in clean_words
        ).strip()

        confidence = self.calculate_confidence(
            clean_words
        )

        usable = self.is_usable(
            clean_text,
            confidence,
        )

        return OCRResult(
            timestamp=timestamp,
            frame=str(frame_path),
            raw_text=raw_text,
            clean_text=clean_text,
            confidence=round(
                confidence,
                2,
            ),
            words=words,
            usable=usable,
        )

    # =========================================================
    # WORD CLEANING
    # =========================================================

    def clean_words(
        self,
        words: list[OCRWord],
    ) -> list[OCRWord]:
        """
        Remove obvious OCR noise.

        We keep:
            - alphabetic words
            - numbers
            - currency/price-like tokens
            - useful mixed alphanumeric tokens

        We remove:
            - standalone punctuation
            - decorative symbols
            - very-low-confidence tokens
        """

        cleaned: list[OCRWord] = []

        for word in words:

            text = word.text.strip()

            if word.confidence < (
                self.minimum_word_confidence
            ):
                continue

            # Remove surrounding decorative punctuation
            normalized = text.strip(
                "\"'`~|_=+-*/\\<>[]{}()"
            )

            if not normalized:
                continue

            # Check whether the token contains useful
            # letters or numbers.
            has_letter = bool(
                re.search(
                    r"[A-Za-z\u0900-\u0D7F]",
                    normalized,
                )
            )

            has_number = bool(
                re.search(
                    r"\d",
                    normalized,
                )
            )

            # Keep normal words and numbers.
            if has_letter or has_number:
                cleaned.append(
                    OCRWord(
                        text=normalized,
                        confidence=word.confidence,
                    )
                )

        return cleaned

    # =========================================================
    # CONFIDENCE
    # =========================================================

    def calculate_confidence(
        self,
        words: list[OCRWord],
    ) -> float:

        if not words:
            return 0.0

        return (
            sum(
                word.confidence
                for word in words
            )
            / len(words)
        )

    # =========================================================
    # USABILITY
    # =========================================================

    def is_usable(
        self,
        text: str,
        confidence: float,
    ) -> bool:

        text = text.strip()

        if not text:
            return False

        if confidence < self.minimum_confidence:
            return False

        meaningful_characters = re.findall(
            r"[A-Za-z0-9\u0900-\u0D7F₹]",
            text,
        )

        if len(meaningful_characters) < 3:
            return False

        return True

    # =========================================================
    # PROCESS ALL FRAMES
    # =========================================================

    def process_frames(
        self,
        frames: list[Path],
        fps: float = 1.0,
    ) -> list[OCRResult]:

        if fps <= 0:
            raise ValueError(
                "FPS must be greater than zero."
            )

        results: list[OCRResult] = []

        for index, frame_path in enumerate(
            frames
        ):

            timestamp = index / fps

            print(
                f"OCR {index + 1}/{len(frames)} "
                f"at {timestamp:.2f}s"
            )

            result = self.process_frame(
                frame_path=frame_path,
                timestamp=timestamp,
            )

            results.append(result)

            if result.usable:
                print(
                    f"  Clean text: "
                    f"{result.clean_text}"
                )

                print(
                    f"  Confidence: "
                    f"{result.confidence:.2f}"
                )

            else:
                print(
                    "  Text: <unusable>"
                )

                print(
                    f"  Raw text: "
                    f"{result.raw_text}"
                )

                print(
                    f"  Confidence: "
                    f"{result.confidence:.2f}"
                )

        return results

    # =========================================================
    # SAVE JSON
    # =========================================================

    def save_results(
        self,
        results: list[OCRResult],
        output_path: str | Path,
    ) -> Path:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "settings": {
                "minimum_confidence": (
                    self.minimum_confidence
                ),
                "minimum_word_confidence": (
                    self.minimum_word_confidence
                ),
            },
            "results": [
                {
                    "timestamp": result.timestamp,
                    "frame": result.frame,
                    "raw_text": result.raw_text,
                    "clean_text": result.clean_text,
                    "confidence": result.confidence,
                    "usable": result.usable,
                    "words": [
                        asdict(word)
                        for word in result.words
                    ],
                }
                for result in results
            ],
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print(
            f"OCR results saved: "
            f"{output_path}"
        )

        return output_path

    # =========================================================
    # SUMMARY
    # =========================================================

    def print_summary(
        self,
        results: list[OCRResult],
    ) -> None:

        usable = [
            result
            for result in results
            if result.usable
        ]

        print()
        print("=" * 70)
        print("OCR SUMMARY")
        print("=" * 70)

        print(
            f"Total frames    : "
            f"{len(results)}"
        )

        print(
            f"Usable frames   : "
            f"{len(usable)}"
        )

        print(
            f"Rejected frames : "
            f"{len(results) - len(usable)}"
        )

        if usable:

            average = (
                sum(
                    result.confidence
                    for result in usable
                )
                / len(usable)
            )

            print(
                f"Average confidence: "
                f"{average:.2f}"
            )

        print("=" * 70)


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    video_name = "audio_test"

    frames_dir = (
        Path("data/frames")
        / video_name
    )

    frames = sorted(
        frames_dir.glob("frame_*.jpg")
    )

    if not frames:
        raise RuntimeError(
            f"No frames found in {frames_dir}"
        )

    processor = OCRProcessor(
        minimum_confidence=30.0,
        minimum_word_confidence=30.0,
    )

    results = processor.process_frames(
        frames=frames,
        fps=1.0,
    )

    output_path = (
        Path("data/outputs")
        / video_name
        / "ocr.json"
    )

    processor.save_results(
        results=results,
        output_path=output_path,
    )

    processor.print_summary(
        results
    )