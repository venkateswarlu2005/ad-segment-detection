from dataclasses import asdict, dataclass
from pathlib import Path
import json

import pytesseract
from PIL import Image


class ShotOCRError(Exception):
    """Raised when shot-level OCR fails."""


@dataclass
class FrameOCR:
    shot_index: int
    position: str
    timestamp: float
    frame_path: str
    text: str
    confidence: float
    usable: bool


@dataclass
class ShotOCR:
    shot_index: int
    start: float
    end: float
    frames: list[FrameOCR]


class ShotOCRProcessor:
    """
    Run OCR on representative frames belonging to shots.

    Input:
        representatives.json
        representative JPG files

    Output:
        shot_ocr.json

    This module does NOT:
        - classify advertisements
        - use a VLM
        - merge advertisements
    """

    def __init__(
        self,
        minimum_word_confidence: float = 30.0,
        minimum_frame_confidence: float = 30.0,
    ):
        self.minimum_word_confidence = (
            minimum_word_confidence
        )

        self.minimum_frame_confidence = (
            minimum_frame_confidence
        )

        print(
            "Shot OCR initialized."
        )

        print(
            f"Minimum word confidence: "
            f"{self.minimum_word_confidence}"
        )

        print(
            f"Minimum frame confidence: "
            f"{self.minimum_frame_confidence}"
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def process(
        self,
        representatives_path: str | Path,
    ) -> list[ShotOCR]:

        representatives_path = Path(
            representatives_path
        )

        if not representatives_path.exists():
            raise ShotOCRError(
                f"Representatives file does not exist: "
                f"{representatives_path}"
            )

        data = self._load_json(
            representatives_path
        )

        if not isinstance(data, list):
            raise ShotOCRError(
                "representatives.json must contain a list."
            )

        results: list[ShotOCR] = []

        total_frames = sum(
            len(shot.get("frames", []))
            for shot in data
        )

        processed = 0

        for shot in data:

            shot_index = int(
                shot["shot_index"]
            )

            start = float(
                shot["start"]
            )

            end = float(
                shot["end"]
            )

            frame_results: list[FrameOCR] = []

            for frame in shot.get(
                "frames",
                [],
            ):

                processed += 1

                position = str(
                    frame["position"]
                )

                timestamp = float(
                    frame["timestamp"]
                )

                frame_path = Path(
                    frame["frame_path"]
                )

                print(
                    f"OCR {processed}/{total_frames} "
                    f"| Shot {shot_index:02d} "
                    f"| {position:6s} "
                    f"| {timestamp:.3f}s"
                )

                result = self._process_frame(
                    shot_index=shot_index,
                    position=position,
                    timestamp=timestamp,
                    frame_path=frame_path,
                )

                frame_results.append(
                    result
                )

                if result.usable:

                    print(
                        f"  Text: "
                        f"{result.text}"
                    )

                    print(
                        f"  Confidence: "
                        f"{result.confidence:.2f}"
                    )

                else:

                    print(
                        f"  Text: <unusable>"
                    )

                    print(
                        f"  Raw confidence: "
                        f"{result.confidence:.2f}"
                    )

            results.append(
                ShotOCR(
                    shot_index=shot_index,
                    start=start,
                    end=end,
                    frames=frame_results,
                )
            )

        return results

    # =========================================================
    # PROCESS ONE FRAME
    # =========================================================

    def _process_frame(
        self,
        shot_index: int,
        position: str,
        timestamp: float,
        frame_path: Path,
    ) -> FrameOCR:

        if not frame_path.exists():

            raise ShotOCRError(
                f"Frame does not exist: "
                f"{frame_path}"
            )

        try:

            image = Image.open(
                frame_path
            )

        except Exception as error:

            raise ShotOCRError(
                f"Could not open frame: "
                f"{frame_path}"
            ) from error

        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
        )

        words = []
        confidences = []

        for index, raw_confidence in enumerate(
            data["conf"]
        ):

            try:

                confidence = float(
                    raw_confidence
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if confidence < 0:
                continue

            text = (
                data["text"][index]
                .strip()
            )

            if not text:
                continue

            if (
                confidence
                >= self.minimum_word_confidence
            ):

                words.append(
                    text
                )

                confidences.append(
                    confidence
                )

        if not words:

            raw_text = pytesseract.image_to_string(
                image
            ).strip()

            fallback_confidences = [
                float(confidence)
                for confidence in data["conf"]
                if isinstance(
                    confidence,
                    (int, float),
                )
                and float(confidence) >= 0
            ]

            if fallback_confidences:

                frame_confidence = sum(
                    fallback_confidences
                ) / len(
                    fallback_confidences
                )

            else:

                frame_confidence = 0.0

            return FrameOCR(
                shot_index=shot_index,
                position=position,
                timestamp=timestamp,
                frame_path=str(
                    frame_path
                ),
                text=raw_text,
                confidence=round(
                    frame_confidence,
                    2,
                ),
                usable=(
                    frame_confidence
                    >= self.minimum_frame_confidence
                    and bool(raw_text)
                ),
            )

        text = " ".join(
            words
        )

        frame_confidence = sum(
            confidences
        ) / len(
            confidences
        )

        usable = (
            frame_confidence
            >= self.minimum_frame_confidence
            and bool(text.strip())
        )

        return FrameOCR(
            shot_index=shot_index,
            position=position,
            timestamp=timestamp,
            frame_path=str(
                frame_path
            ),
            text=text,
            confidence=round(
                frame_confidence,
                2,
            ),
            usable=usable,
        )

    # =========================================================
    # LOAD JSON
    # =========================================================

    @staticmethod
    def _load_json(
        path: Path,
    ):

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(
                    file
                )

        except json.JSONDecodeError as error:

            raise ShotOCRError(
                f"Invalid JSON: {path}"
            ) from error

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    @staticmethod
    def save_results(
        results: list[ShotOCR],
        output_path: str | Path,
    ) -> Path:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = [
            asdict(result)
            for result in results
        ]

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
            f"Shot OCR results saved: "
            f"{output_path}"
        )

        return output_path

    # =========================================================
    # SUMMARY
    # =========================================================

    @staticmethod
    def print_summary(
        results: list[ShotOCR],
    ) -> None:

        total_frames = 0
        usable_frames = 0

        all_confidences = []

        for shot in results:

            for frame in shot.frames:

                total_frames += 1

                if frame.usable:
                    usable_frames += 1

                all_confidences.append(
                    frame.confidence
                )

        rejected_frames = (
            total_frames
            - usable_frames
        )

        if all_confidences:

            average_confidence = (
                sum(all_confidences)
                / len(all_confidences)
            )

        else:

            average_confidence = 0.0

        print()
        print("=" * 70)
        print("SHOT OCR SUMMARY")
        print("=" * 70)

        print(
            f"Shots           : "
            f"{len(results)}"
        )

        print(
            f"Total frames    : "
            f"{total_frames}"
        )

        print(
            f"Usable frames   : "
            f"{usable_frames}"
        )

        print(
            f"Rejected frames : "
            f"{rejected_frames}"
        )

        print(
            f"Average confidence: "
            f"{average_confidence:.2f}"
        )

        print("=" * 70)


# =============================================================
# COMMAND-LINE TEST
# =============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python -m src.shot_ocr "
            "<video_path>"
        )

        sys.exit(1)

    video_path = Path(
        sys.argv[1]
    )

    representatives_path = (
        Path("data/outputs")
        / video_path.stem
        / "representatives.json"
    )

    output_path = (
        Path("data/outputs")
        / video_path.stem
        / "shot_ocr.json"
    )

    processor = ShotOCRProcessor(
        minimum_word_confidence=30.0,
        minimum_frame_confidence=30.0,
    )

    try:

        results = processor.process(
            representatives_path
        )

        processor.print_summary(
            results
        )

        processor.save_results(
            results,
            output_path,
        )

    except ShotOCRError as error:

        print()
        print(
            "SHOT OCR FAILED"
        )

        print(error)

        sys.exit(1)