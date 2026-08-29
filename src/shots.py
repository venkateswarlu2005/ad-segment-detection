from dataclasses import asdict, dataclass
from pathlib import Path
import json

import cv2


class ShotExtractionError(Exception):
    """Raised when representative frame extraction fails."""


@dataclass
class RepresentativeFrame:
    shot_index: int
    position: str
    timestamp: float
    frame_path: str


@dataclass
class ShotFrameSet:
    shot_index: int
    start: float
    end: float
    frames: list[RepresentativeFrame]


class ShotFrameExtractor:
    """
    Extract representative frames from detected shots.

    For each shot we currently extract:
        - early frame
        - middle frame
        - late frame

    This module does NOT:
        - perform OCR
        - perform VLM classification
        - detect advertisements
    """

    def __init__(
        self,
        output_dir: str | Path = "data/frames/shots",
    ):
        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def extract(
        self,
        video_path: str | Path,
        shots_path: str | Path,
    ) -> list[ShotFrameSet]:

        video_path = Path(video_path)
        shots_path = Path(shots_path)

        self._validate_video(
            video_path
        )

        self._validate_shots_file(
            shots_path
        )

        shots = self._load_shots(
            shots_path
        )

        if not shots:
            raise ShotExtractionError(
                "No shots found in scenes.json."
            )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise ShotExtractionError(
                f"Could not open video: "
                f"{video_path}"
            )

        video_name = video_path.stem

        video_output_dir = (
            self.output_dir
            / video_name
        )

        video_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results: list[ShotFrameSet] = []

        try:

            for shot_index, shot in enumerate(
                shots,
                start=1,
            ):

                result = self._extract_shot_frames(
                    capture=capture,
                    shot_index=shot_index,
                    start=float(shot["start"]),
                    end=float(shot["end"]),
                    output_dir=video_output_dir,
                )

                results.append(result)

        finally:

            capture.release()

        return results

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_video(
        video_path: Path,
    ) -> None:

        if not video_path.exists():
            raise ShotExtractionError(
                f"Video does not exist: "
                f"{video_path}"
            )

        if not video_path.is_file():
            raise ShotExtractionError(
                f"Video path is not a file: "
                f"{video_path}"
            )

    @staticmethod
    def _validate_shots_file(
        shots_path: Path,
    ) -> None:

        if not shots_path.exists():
            raise ShotExtractionError(
                f"Scenes file does not exist: "
                f"{shots_path}"
            )

        if not shots_path.is_file():
            raise ShotExtractionError(
                f"Scenes path is not a file: "
                f"{shots_path}"
            )

    # =========================================================
    # LOAD SHOTS
    # =========================================================

    @staticmethod
    def _load_shots(
        shots_path: Path,
    ) -> list[dict]:

        try:

            with shots_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

        except json.JSONDecodeError as error:

            raise ShotExtractionError(
                f"Invalid JSON in "
                f"{shots_path}"
            ) from error

        shots = data.get(
            "shots"
        )

        if not isinstance(
            shots,
            list,
        ):

            raise ShotExtractionError(
                "The scenes file does not contain "
                "a valid 'shots' list."
            )

        return shots

    # =========================================================
    # EXTRACT FRAMES FOR ONE SHOT
    # =========================================================

    def _extract_shot_frames(
        self,
        capture,
        shot_index: int,
        start: float,
        end: float,
        output_dir: Path,
    ) -> ShotFrameSet:

        if end <= start:
            raise ShotExtractionError(
                f"Invalid shot {shot_index}: "
                f"{start} → {end}"
            )

        duration = end - start

        # -----------------------------------------------------
        # Choose three timestamps.
        #
        # We keep them slightly inside the shot rather than
        # exactly on the boundaries to avoid accidentally
        # capturing the neighboring shot.
        # -----------------------------------------------------

        margin = min(
            0.10,
            duration * 0.10,
        )

        early_time = start + margin

        middle_time = (
            start + end
        ) / 2.0

        late_time = end - margin

        timestamps = [
            ("early", early_time),
            ("middle", middle_time),
            ("late", late_time),
        ]

        frames: list[RepresentativeFrame] = []

        for position, timestamp in timestamps:

            frame_path = (
                output_dir
                / (
                    f"shot_{shot_index:02d}"
                    f"_{position}.jpg"
                )
            )

            success = self._save_frame(
                capture=capture,
                timestamp=timestamp,
                output_path=frame_path,
            )

            if not success:

                raise ShotExtractionError(
                    f"Could not extract frame "
                    f"at {timestamp:.3f}s "
                    f"for shot {shot_index}."
                )

            frames.append(
                RepresentativeFrame(
                    shot_index=shot_index,
                    position=position,
                    timestamp=round(
                        timestamp,
                        3,
                    ),
                    frame_path=str(
                        frame_path
                    ),
                )
            )

        return ShotFrameSet(
            shot_index=shot_index,
            start=round(
                start,
                3,
            ),
            end=round(
                end,
                3,
            ),
            frames=frames,
        )

    # =========================================================
    # SAVE ONE FRAME
    # =========================================================

    @staticmethod
    def _save_frame(
        capture,
        timestamp: float,
        output_path: Path,
    ) -> bool:

        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000.0,
        )

        success, frame = capture.read()

        if not success:
            return False

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return bool(
            cv2.imwrite(
                str(output_path),
                frame,
            )
        )

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    def save_results(
        self,
        results: list[ShotFrameSet],
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
            f"Representative frame "
            f"metadata saved: {output_path}"
        )

        return output_path

    # =========================================================
    # DISPLAY
    # =========================================================

    @staticmethod
    def print_results(
        results: list[ShotFrameSet],
    ) -> None:

        print()
        print("=" * 70)
        print("REPRESENTATIVE FRAME EXTRACTION")
        print("=" * 70)

        print(
            f"Shots processed : "
            f"{len(results)}"
        )

        total_frames = sum(
            len(result.frames)
            for result in results
        )

        print(
            f"Frames saved    : "
            f"{total_frames}"
        )

        print()

        for result in results:

            print(
                f"Shot {result.shot_index:02d}: "
                f"{result.start:.3f}s → "
                f"{result.end:.3f}s"
            )

            for frame in result.frames:

                print(
                    f"  {frame.position:6s} "
                    f"{frame.timestamp:7.3f}s "
                    f"→ "
                    f"{frame.frame_path}"
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
            "  python -m src.shots "
            "<video_path>"
        )

        sys.exit(1)

    video_path = Path(
        sys.argv[1]
    )

    scenes_path = (
        Path("data/outputs")
        / video_path.stem
        / "scenes.json"
    )

    output_dir = (
        Path("data/frames/shots")
        / video_path.stem
    )

    output_json = (
        Path("data/outputs")
        / video_path.stem
        / "representatives.json"
    )

    extractor = ShotFrameExtractor(
        output_dir=output_dir
    )

    try:

        results = extractor.extract(
            video_path=video_path,
            shots_path=scenes_path,
        )

        extractor.print_results(
            results
        )

        extractor.save_results(
            results,
            output_json,
        )

    except (
        ShotExtractionError,
        ValueError,
    ) as error:

        print()
        print(
            "SHOT FRAME EXTRACTION FAILED"
        )

        print(error)

        sys.exit(1)