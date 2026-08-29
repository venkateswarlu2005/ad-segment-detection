from dataclasses import asdict, dataclass
from pathlib import Path
import json

import cv2


class SceneDetectionError(Exception):
    """Raised when scene detection fails."""


@dataclass
class SceneBoundary:
    """
    A detected boundary between two shots.
    """

    timestamp: float
    difference: float


@dataclass
class Shot:
    """
    A continuous visual shot in the video.
    """

    start: float
    end: float


@dataclass
class SceneDetectionResult:
    """
    Complete scene detection result.
    """

    video: str
    duration: float
    sample_fps: float
    threshold: float
    boundaries: list[SceneBoundary]
    shots: list[Shot]


class SceneDetector:
    """
    Detect scene changes using frame-to-frame
    visual differences.

    This is a deterministic baseline.

    It does NOT:
        - detect advertisements
        - perform OCR
        - use a VLM
        - classify content
    """

    def __init__(
        self,
        sample_fps: float = 5.0,
        threshold: float = 25.0,
    ):
        if sample_fps <= 0:
            raise ValueError(
                "sample_fps must be greater than zero."
            )

        if threshold <= 0:
            raise ValueError(
                "threshold must be greater than zero."
            )

        self.sample_fps = sample_fps
        self.threshold = threshold

    # =========================================================
    # PUBLIC API
    # =========================================================

    def detect(
        self,
        video_path: str | Path,
    ) -> SceneDetectionResult:

        video_path = Path(video_path)

        if not video_path.exists():
            raise SceneDetectionError(
                f"Video does not exist: {video_path}"
            )

        if not video_path.is_file():
            raise SceneDetectionError(
                f"Video path is not a file: {video_path}"
            )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise SceneDetectionError(
                f"Could not open video: {video_path}"
            )

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

        if fps <= 0:
            capture.release()

            raise SceneDetectionError(
                "Could not determine video FPS."
            )

        if frame_count <= 0:
            capture.release()

            raise SceneDetectionError(
                "Could not determine video frame count."
            )

        duration = frame_count / fps

        sample_interval = 1.0 / self.sample_fps

        next_sample_time = 0.0

        previous_gray = None

        boundaries: list[SceneBoundary] = []

        while True:

            success, frame = capture.read()

            if not success:
                break

            current_frame_number = (
                capture.get(
                    cv2.CAP_PROP_POS_FRAMES
                )
            )

            current_time = (
                current_frame_number / fps
            )

            if current_time < next_sample_time:
                continue

            next_sample_time += sample_interval

            # ---------------------------------------------
            # Resize for faster comparison
            # ---------------------------------------------

            resized = cv2.resize(
                frame,
                (320, 180),
            )

            gray = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2GRAY,
            )

            # ---------------------------------------------
            # Compare with previous sample
            # ---------------------------------------------

            if previous_gray is not None:

                difference = cv2.absdiff(
                    previous_gray,
                    gray,
                )

                mean_difference = float(
                    difference.mean()
                )

                if (
                    mean_difference
                    >= self.threshold
                ):

                    boundaries.append(
                        SceneBoundary(
                            timestamp=round(
                                current_time,
                                3,
                            ),
                            difference=round(
                                mean_difference,
                                3,
                            ),
                        )
                    )

            previous_gray = gray

        capture.release()

        shots = self._build_shots(
            duration=duration,
            boundaries=boundaries,
        )

        return SceneDetectionResult(
            video=str(video_path),
            duration=round(
                duration,
                3,
            ),
            sample_fps=self.sample_fps,
            threshold=self.threshold,
            boundaries=boundaries,
            shots=shots,
        )

    # =========================================================
    # BUILD SHOTS
    # =========================================================

    @staticmethod
    def _build_shots(
        duration: float,
        boundaries: list[SceneBoundary],
    ) -> list[Shot]:

        if duration <= 0:
            return []

        timestamps = sorted(
            {
                boundary.timestamp
                for boundary in boundaries
                if 0 < boundary.timestamp < duration
            }
        )

        shots: list[Shot] = []

        start = 0.0

        for timestamp in timestamps:

            if timestamp <= start:
                continue

            shots.append(
                Shot(
                    start=round(
                        start,
                        3,
                    ),
                    end=round(
                        timestamp,
                        3,
                    ),
                )
            )

            start = timestamp

        if start < duration:

            shots.append(
                Shot(
                    start=round(
                        start,
                        3,
                    ),
                    end=round(
                        duration,
                        3,
                    ),
                )
            )

        return shots

    # =========================================================
    # DISPLAY
    # =========================================================

    @staticmethod
    def print_result(
        result: SceneDetectionResult,
    ) -> None:

        print()
        print("=" * 70)
        print("SCENE DETECTION")
        print("=" * 70)

        print(
            f"Video       : "
            f"{result.video}"
        )

        print(
            f"Duration    : "
            f"{result.duration:.3f}s"
        )

        print(
            f"Sample FPS  : "
            f"{result.sample_fps}"
        )

        print(
            f"Threshold   : "
            f"{result.threshold}"
        )

        print()

        print(
            f"Scene cuts  : "
            f"{len(result.boundaries)}"
        )

        print(
            f"Shots       : "
            f"{len(result.shots)}"
        )

        print()

        if result.boundaries:

            print("BOUNDARIES")
            print("-" * 70)

            for boundary in result.boundaries:

                print(
                    f"{boundary.timestamp:8.3f}s"
                    f" | difference="
                    f"{boundary.difference:.3f}"
                )

        else:

            print(
                "No scene boundaries detected."
            )

        print()

        print("SHOTS")
        print("-" * 70)

        for index, shot in enumerate(
            result.shots,
            start=1,
        ):

            print(
                f"Shot {index:02d}: "
                f"{shot.start:.3f}s → "
                f"{shot.end:.3f}s"
            )

        print("=" * 70)

    # =========================================================
    # SAVE JSON
    # =========================================================

    @staticmethod
    def save_result(
        result: SceneDetectionResult,
        output_path: str | Path,
    ) -> Path:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = asdict(
            result
        )

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
            f"Scene result saved: "
            f"{output_path}"
        )

        return output_path


# =============================================================
# COMMAND-LINE TEST
# =============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python -m src.scene "
            "<video_path>"
        )

        sys.exit(1)

    video_path = sys.argv[1]

    detector = SceneDetector(
        sample_fps=5.0,
        threshold=25.0,
    )

    try:

        result = detector.detect(
            video_path
        )

        detector.print_result(
            result
        )

        output_path = (
            Path("data/outputs")
            / Path(video_path).stem
            / "scenes.json"
        )

        detector.save_result(
            result,
            output_path,
        )

    except (
        SceneDetectionError,
        ValueError,
    ) as error:

        print()
        print(
            "SCENE DETECTION FAILED"
        )

        print(error)

        sys.exit(1)