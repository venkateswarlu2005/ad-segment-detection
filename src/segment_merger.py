import json
import sys
from pathlib import Path


class SegmentMergeError(Exception):
    """Raised when segment merging fails."""


class SegmentMerger:
    """
    Convert shot-level advertisement scores into
    continuous advertisement segments.

    Input:
        data/outputs/<video>/ad_scores.json

    Output:
        data/outputs/<video>/segments.json
    """

    def __init__(
        self,
        advertisement_threshold: float = 0.70,
        bridge_threshold: float = 0.40,
        max_bridge_gap: float = 1.5,
    ):
        self.advertisement_threshold = (
            advertisement_threshold
        )

        self.bridge_threshold = (
            bridge_threshold
        )

        self.max_bridge_gap = (
            max_bridge_gap
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def merge_video(
        self,
        video_path: str | Path,
    ) -> dict:

        video_path = Path(
            video_path
        )

        video_name = (
            video_path.stem
        )

        output_dir = (
            Path("data/outputs")
            / video_name
        )

        input_path = (
            output_dir
            / "ad_scores.json"
        )

        if not input_path.exists():

            raise SegmentMergeError(
                f"Ad scores not found: "
                f"{input_path}"
            )

        data = self._load_json(
            input_path
        )

        shots = data.get(
            "shots",
            [],
        )

        if not shots:

            raise SegmentMergeError(
                "No shot scores found."
            )

        shots = sorted(
            shots,
            key=lambda shot: float(
                shot["start"]
            ),
        )

        print()
        print("=" * 70)
        print("SEGMENT MERGING")
        print("=" * 70)

        print(
            f"Video : {video_path}"
        )

        print(
            f"Shots : {len(shots)}"
        )

        print(
            f"Advertisement threshold : "
            f"{self.advertisement_threshold:.2f}"
        )

        print(
            f"Bridge threshold         : "
            f"{self.bridge_threshold:.2f}"
        )

        print(
            f"Maximum bridge gap       : "
            f"{self.max_bridge_gap:.2f}s"
        )

        print("=" * 70)

        # -----------------------------------------------------
        # STEP 1
        # Mark every shot as AD / UNCERTAIN / ORGANIC
        # -----------------------------------------------------

        states = []

        print()
        print("SHOT STATES")
        print("-" * 70)

        for shot in shots:

            score = float(
                shot.get(
                    "ad_score",
                    0.0,
                )
            )

            if score >= (
                self.advertisement_threshold
            ):

                state = "advertisement"

            elif score >= (
                self.bridge_threshold
            ):

                state = "uncertain"

            else:

                state = "organic_content"

            state_item = {
                "shot": shot,
                "state": state,
                "bridged": False,
            }

            states.append(
                state_item
            )

            print(
                f"Shot "
                f"{int(shot['shot_index']):02d} "
                f"| "
                f"{shot['start']:.3f}s → "
                f"{shot['end']:.3f}s "
                f"| score="
                f"{score:.3f} "
                f"| {state}"
            )

        # -----------------------------------------------------
        # STEP 2
        # Bridge uncertain shots between advertisements
        # -----------------------------------------------------

        print()
        print("BRIDGING")
        print("-" * 70)

        for index in range(
            1,
            len(states) - 1,
        ):

            current = states[index]
            previous = states[index - 1]
            following = states[index + 1]

            if current["state"] != "uncertain":
                continue

            if (
                previous["state"]
                != "advertisement"
            ):
                continue

            if (
                following["state"]
                != "advertisement"
            ):
                continue

            current_shot = current["shot"]
            previous_shot = previous["shot"]
            following_shot = following["shot"]

            gap_before = (
                float(current_shot["start"])
                - float(previous_shot["end"])
            )

            gap_after = (
                float(following_shot["start"])
                - float(current_shot["end"])
            )

            if (
                gap_before
                <= self.max_bridge_gap
                and
                gap_after
                <= self.max_bridge_gap
            ):

                current["bridged"] = True

                print(
                    f"Shot "
                    f"{int(current_shot['shot_index']):02d} "
                    f"bridged between Shots "
                    f"{int(previous_shot['shot_index']):02d} "
                    f"and "
                    f"{int(following_shot['shot_index']):02d}"
                )

        # -----------------------------------------------------
        # STEP 3
        # Determine which shots belong to ad segments
        # -----------------------------------------------------

        ad_shots = []

        for state_item in states:

            if (
                state_item["state"]
                == "advertisement"
                or state_item["bridged"]
            ):

                ad_shots.append(
                    state_item
                )

        # -----------------------------------------------------
        # STEP 4
        # Merge consecutive ad shots
        # -----------------------------------------------------

        segments = []

        current_segment = None

        for state_item in states:

            is_ad = (
                state_item["state"]
                == "advertisement"
                or state_item["bridged"]
            )

            shot = state_item["shot"]

            if not is_ad:

                if current_segment:

                    segments.append(
                        self._finalize_segment(
                            current_segment
                        )
                    )

                    current_segment = None

                continue

            if current_segment is None:

                current_segment = {
                    "start": float(
                        shot["start"]
                    ),
                    "end": float(
                        shot["end"]
                    ),
                    "shots": [
                        state_item
                    ],
                }

                continue

            previous_end = float(
                current_segment["end"]
            )

            current_start = float(
                shot["start"]
            )

            gap = (
                current_start
                - previous_end
            )

            if gap <= self.max_bridge_gap:

                current_segment["end"] = (
                    float(
                        shot["end"]
                    )
                )

                current_segment[
                    "shots"
                ].append(
                    state_item
                )

            else:

                segments.append(
                    self._finalize_segment(
                        current_segment
                    )
                )

                current_segment = {
                    "start": float(
                        shot["start"]
                    ),
                    "end": float(
                        shot["end"]
                    ),
                    "shots": [
                        state_item
                    ],
                }

        if current_segment:

            segments.append(
                self._finalize_segment(
                    current_segment
                )
            )

        # -----------------------------------------------------
        # STEP 5
        # Build final output
        # -----------------------------------------------------

        output = {
            "schema_version": "1.0",
            "source": {
                "video": str(
                    video_path
                ),
                "ad_scores": str(
                    input_path
                ),
            },
            "method": {
                "advertisement_threshold": (
                    self.advertisement_threshold
                ),
                "bridge_threshold": (
                    self.bridge_threshold
                ),
                "max_bridge_gap": (
                    self.max_bridge_gap
                ),
            },
            "segments": segments,
            "stats": {
                "total_shots": len(
                    shots
                ),
                "total_segments": len(
                    segments
                ),
            },
        }

        output_path = (
            output_dir
            / "segments.json"
        )

        self._save_json(
            output,
            output_path,
        )

        self._print_segments(
            segments
        )

        print()
        print(
            f"Segments saved: "
            f"{output_path}"
        )

        return output

    # =========================================================
    # FINALIZE SEGMENT
    # =========================================================

    def _finalize_segment(
        self,
        segment: dict,
    ) -> dict:

        shot_items = segment[
            "shots"
        ]

        scores = []

        shot_indices = []

        evidence = []

        for item in shot_items:

            shot = item["shot"]

            score = float(
                shot.get(
                    "ad_score",
                    0.0,
                )
            )

            scores.append(
                score
            )

            shot_indices.append(
                int(
                    shot["shot_index"]
                )
            )

            evidence.append(
                {
                    "shot_index": int(
                        shot["shot_index"]
                    ),
                    "start": float(
                        shot["start"]
                    ),
                    "end": float(
                        shot["end"]
                    ),
                    "ad_score": round(
                        score,
                        3,
                    ),
                    "classification": (
                        shot.get(
                            "classification",
                            "uncertain",
                        )
                    ),
                    "bridged": bool(
                        item["bridged"]
                    ),
                    "vlm_classification": (
                        shot.get(
                            "vlm_classification"
                        )
                    ),
                    "vlm_confidence": (
                        shot.get(
                            "vlm_confidence"
                        )
                    ),
                    "evidence_labels": (
                        shot.get(
                            "evidence_labels",
                            [],
                        )
                    ),
                    "ocr_text": (
                        shot.get(
                            "ocr_text",
                            "",
                        )
                    ),
                    "transcript": (
                        shot.get(
                            "transcript",
                            "",
                        )
                    ),
                }
            )

        # Weighted toward the strongest evidence while
        # still considering the entire segment.

        if scores:

            strongest = max(
                scores
            )

            average = (
                sum(scores)
                / len(scores)
            )

            confidence = (
                0.7 * strongest
                + 0.3 * average
            )

        else:

            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        return {
            "start": round(
                float(
                    segment["start"]
                ),
                3,
            ),
            "end": round(
                float(
                    segment["end"]
                ),
                3,
            ),
            "duration": round(
                float(
                    segment["end"]
                )
                - float(
                    segment["start"]
                ),
                3,
            ),
            "type": "advertisement",
            "confidence": round(
                confidence,
                3,
            ),
            "shots": shot_indices,
            "evidence": evidence,
        }

    # =========================================================
    # PRINT
    # =========================================================

    @staticmethod
    def _print_segments(
        segments: list,
    ) -> None:

        print()
        print("=" * 70)
        print("FINAL AD SEGMENTS")
        print("=" * 70)

        if not segments:

            print(
                "No advertisement segments detected."
            )

            print("=" * 70)

            return

        for index, segment in enumerate(
            segments,
            start=1,
        ):

            print(
                f"Segment {index:02d}: "
                f"{segment['start']:.3f}s → "
                f"{segment['end']:.3f}s "
                f"("
                f"{segment['duration']:.3f}s"
                ")"
            )

            print(
                f"  Type       : "
                f"{segment['type']}"
            )

            print(
                f"  Confidence : "
                f"{segment['confidence']:.3f}"
            )

            print(
                f"  Shots      : "
                f"{segment['shots']}"
            )

        print("=" * 70)

    # =========================================================
    # JSON HELPERS
    # =========================================================

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict:

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(
                    file
                )

        except json.JSONDecodeError as error:

            raise SegmentMergeError(
                f"Invalid JSON: {path}"
            ) from error

        except OSError as error:

            raise SegmentMergeError(
                f"Could not read: {path}"
            ) from error

    @staticmethod
    def _save_json(
        data: dict,
        path: Path,
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )


# =============================================================
# COMMAND LINE
# =============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python -m src.segment_merger "
            "<video_path>"
        )

        sys.exit(1)

    video_path = Path(
        sys.argv[1]
    )

    merger = SegmentMerger()

    try:

        merger.merge_video(
            video_path
        )

    except SegmentMergeError as error:

        print()
        print(
            "SEGMENT MERGING FAILED"
        )

        print(error)

        sys.exit(1)