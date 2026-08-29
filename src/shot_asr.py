from dataclasses import asdict, dataclass
from pathlib import Path
import json
import sys


class ShotASRError(Exception):
    """Raised when ASR-to-shot alignment fails."""


@dataclass
class AlignedASRSegment:
    start: float
    end: float
    text: str
    overlap_start: float
    overlap_end: float
    overlap_duration: float


@dataclass
class ShotASR:
    shot_index: int
    start: float
    end: float
    duration: float
    transcript: list[AlignedASRSegment]
    combined_text: str


class ShotASRAligner:
    """
    Align Whisper transcript segments with detected shots.

    Input:
        transcript.json
        scenes.json

    Output:
        shot_asr.json

    Alignment rule:
        An ASR segment belongs to a shot when the two
        time intervals overlap.

    Example:

        Shot:
            4.421 -> 6.840

        ASR:
            4.000 -> 6.000

        Overlap:
            4.421 -> 6.000
    """

    def __init__(
        self,
        minimum_overlap: float = 0.0,
    ):
        if minimum_overlap < 0:
            raise ValueError(
                "minimum_overlap cannot be negative."
            )

        self.minimum_overlap = minimum_overlap

    # =========================================================
    # PUBLIC API
    # =========================================================

    def align(
        self,
        transcript_path: str | Path,
        scenes_path: str | Path,
    ) -> list[ShotASR]:

        transcript_path = Path(
            transcript_path
        )

        scenes_path = Path(
            scenes_path
        )

        transcript_data = self._load_json(
            transcript_path
        )

        scenes_data = self._load_json(
            scenes_path
        )

        asr_segments = self._extract_asr_segments(
            transcript_data
        )

        shots = self._extract_shots(
            scenes_data
        )

        results: list[ShotASR] = []

        for shot_index, shot in enumerate(
            shots,
            start=1,
        ):

            shot_start = float(
                shot["start"]
            )

            shot_end = float(
                shot["end"]
            )

            aligned_segments = []

            for segment in asr_segments:

                asr_start = float(
                    segment["start"]
                )

                asr_end = float(
                    segment["end"]
                )

                overlap_start = max(
                    shot_start,
                    asr_start,
                )

                overlap_end = min(
                    shot_end,
                    asr_end,
                )

                overlap_duration = (
                    overlap_end
                    - overlap_start
                )

                if (
                    overlap_duration
                    <= self.minimum_overlap
                ):
                    continue

                aligned_segments.append(
                    AlignedASRSegment(
                        start=asr_start,
                        end=asr_end,
                        text=str(
                            segment["text"]
                        ).strip(),
                        overlap_start=round(
                            overlap_start,
                            3,
                        ),
                        overlap_end=round(
                            overlap_end,
                            3,
                        ),
                        overlap_duration=round(
                            overlap_duration,
                            3,
                        ),
                    )
                )

            # -------------------------------------------------
            # Sort speech chronologically.
            # -------------------------------------------------

            aligned_segments.sort(
                key=lambda segment:
                segment.start
            )

            combined_text = " ".join(
                segment.text
                for segment in aligned_segments
                if segment.text
            )

            results.append(
                ShotASR(
                    shot_index=shot_index,
                    start=round(
                        shot_start,
                        3,
                    ),
                    end=round(
                        shot_end,
                        3,
                    ),
                    duration=round(
                        shot_end - shot_start,
                        3,
                    ),
                    transcript=aligned_segments,
                    combined_text=combined_text,
                )
            )

        return results

    # =========================================================
    # LOAD JSON
    # =========================================================

    @staticmethod
    def _load_json(
        path: Path,
    ):

        if not path.exists():
            raise ShotASRError(
                f"File does not exist: {path}"
            )

        if not path.is_file():
            raise ShotASRError(
                f"Path is not a file: {path}"
            )

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(file)

        except json.JSONDecodeError as error:

            raise ShotASRError(
                f"Invalid JSON: {path}"
            ) from error

    # =========================================================
    # EXTRACT ASR SEGMENTS
    # =========================================================

    @staticmethod
    def _extract_asr_segments(
        data,
    ) -> list[dict]:

        segments = data.get(
            "segments"
        )

        if not isinstance(
            segments,
            list,
        ):

            raise ShotASRError(
                "transcript.json does not contain "
                "a valid 'segments' list."
            )

        return segments

    # =========================================================
    # EXTRACT SHOTS
    # =========================================================

    @staticmethod
    def _extract_shots(
        data,
    ) -> list[dict]:

        shots = data.get(
            "shots"
        )

        if not isinstance(
            shots,
            list,
        ):

            raise ShotASRError(
                "scenes.json does not contain "
                "a valid 'shots' list."
            )

        return shots

    # =========================================================
    # DISPLAY
    # =========================================================

    @staticmethod
    def print_results(
        results: list[ShotASR],
    ) -> None:

        print()
        print("=" * 70)
        print("ASR → SHOT ALIGNMENT")
        print("=" * 70)

        print(
            f"Shots processed: "
            f"{len(results)}"
        )

        print()

        for result in results:

            print(
                f"Shot {result.shot_index:02d}: "
                f"{result.start:.3f}s → "
                f"{result.end:.3f}s"
            )

            if not result.transcript:

                print(
                    "  Speech: <none>"
                )

                continue

            for segment in result.transcript:

                print(
                    f"  ASR "
                    f"{segment.start:.3f}s → "
                    f"{segment.end:.3f}s "
                    f"| overlap="
                    f"{segment.overlap_duration:.3f}s"
                )

                print(
                    f"    {segment.text}"
                )

            print(
                f"  Combined: "
                f"{result.combined_text}"
            )

        print("=" * 70)

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    @staticmethod
    def save_results(
        results: list[ShotASR],
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
            f"Shot ASR results saved: "
            f"{output_path}"
        )

        return output_path


# =============================================================
# COMMAND-LINE ENTRY POINT
# =============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python -m src.shot_asr "
            "<video_path>"
        )

        sys.exit(1)

    video_path = Path(
        sys.argv[1]
    )

    video_name = video_path.stem

    transcript_path = (
        Path("data/outputs")
        / video_name
        / "transcript.json"
    )

    scenes_path = (
        Path("data/outputs")
        / video_name
        / "scenes.json"
    )

    output_path = (
        Path("data/outputs")
        / video_name
        / "shot_asr.json"
    )

    aligner = ShotASRAligner()

    try:

        results = aligner.align(
            transcript_path=transcript_path,
            scenes_path=scenes_path,
        )

        aligner.print_results(
            results
        )

        aligner.save_results(
            results,
            output_path,
        )

    except (
        ShotASRError,
        ValueError,
    ) as error:

        print()
        print(
            "SHOT ASR ALIGNMENT FAILED"
        )

        print(error)

        sys.exit(1)