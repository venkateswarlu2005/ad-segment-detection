import json
import sys
from pathlib import Path


class LongSegmentMergerError(Exception):
    """Raised when long-form segment merging fails."""


# =============================================================
# CONFIGURATION
# =============================================================

OUTPUT_ROOT = Path("data") / "outputs"

# Advertisements separated by <= this gap are merged.
MERGE_GAP_SECONDS = 5.0

# VLM classifications treated as advertisements.
ADVERTISEMENT_LABELS = {
    "advertisement",
    "ad",
    "commercial",
}


# =============================================================
# LONG-FORM SEGMENT MERGER
# =============================================================

class LongSegmentMerger:

    def __init__(
        self,
        merge_gap_seconds: float = MERGE_GAP_SECONDS,
    ):
        self.merge_gap_seconds = max(
            0.0,
            float(merge_gap_seconds),
        )

    # =========================================================
    # MAIN
    # =========================================================

    def merge(
        self,
        video_path: str | Path,
    ) -> dict:

        video_path = Path(video_path)

        if not video_path.exists():
            raise LongSegmentMergerError(
                f"Video not found: {video_path}"
            )

        output_dir = (
            OUTPUT_ROOT
            / video_path.stem
        )

        vlm_path = (
            output_dir
            / "long_vlm_results.json"
        )

        candidates_path = (
            output_dir
            / "long_candidates.json"
        )

        evidence_path = (
            output_dir
            / "long_evidence.json"
        )

        # -----------------------------------------------------
        # Validate input files
        # -----------------------------------------------------

        if not vlm_path.exists():
            raise LongSegmentMergerError(
                f"Missing long_vlm_results.json: "
                f"{vlm_path}"
            )

        if not candidates_path.exists():
            raise LongSegmentMergerError(
                f"Missing long_candidates.json: "
                f"{candidates_path}"
            )

        if not evidence_path.exists():
            raise LongSegmentMergerError(
                f"Missing long_evidence.json: "
                f"{evidence_path}"
            )

        # -----------------------------------------------------
        # Load JSON files
        # -----------------------------------------------------

        vlm_data = self._load_json(
            vlm_path
        )

        candidates_data = self._load_json(
            candidates_path
        )

        evidence_data = self._load_json(
            evidence_path
        )

        # IMPORTANT:
        # long_vlm_results.json uses "candidates",
        # not "results".
        vlm_results = vlm_data.get(
            "candidates",
            [],
        )

        candidates = candidates_data.get(
            "candidates",
            [],
        )

        evidence_regions = evidence_data.get(
            "regions",
            [],
        )

        if not vlm_results:
            raise LongSegmentMergerError(
                "No VLM candidates found in "
                "long_vlm_results.json"
            )

        # -----------------------------------------------------
        # Build candidate lookup
        # -----------------------------------------------------

        candidate_by_index = {}

        for candidate in candidates:

            candidate_index = candidate.get(
                "candidate_index"
            )

            if candidate_index is None:
                continue

            try:
                candidate_index = int(
                    candidate_index
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            candidate_by_index[
                candidate_index
            ] = candidate

        # -----------------------------------------------------
        # Build evidence lookup
        # -----------------------------------------------------

        evidence_by_region = {}

        for region in evidence_regions:

            region_index = region.get(
                "region_index"
            )

            if region_index is None:
                continue

            try:
                region_index = int(
                    region_index
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            evidence_by_region[
                region_index
            ] = region

        # -----------------------------------------------------
        # Extract advertisements
        # -----------------------------------------------------

        advertisement_segments = []

        for vlm_result in vlm_results:

            classification = str(
                vlm_result.get(
                    "classification",
                    "",
                )
            ).strip().lower()

            # Ignore organic / uncertain results.
            if classification not in ADVERTISEMENT_LABELS:
                continue

            candidate_index = (
                vlm_result.get(
                    "candidate_index"
                )
            )

            candidate = None

            if candidate_index is not None:

                try:
                    candidate_index = int(
                        candidate_index
                    )

                    candidate = (
                        candidate_by_index.get(
                            candidate_index
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    candidate_index = None

            # -------------------------------------------------
            # Determine boundaries
            # -------------------------------------------------

            start = self._safe_float(
                vlm_result.get(
                    "start"
                ),
                None,
            )

            end = self._safe_float(
                vlm_result.get(
                    "end"
                ),
                None,
            )

            # Fall back to candidate boundaries.
            if candidate is not None:

                if start is None:
                    start = self._safe_float(
                        candidate.get(
                            "start"
                        ),
                        None,
                    )

                if end is None:
                    end = self._safe_float(
                        candidate.get(
                            "end"
                        ),
                        None,
                    )

            if start is None or end is None:
                continue

            if end <= start:
                continue

            # -------------------------------------------------
            # Metadata
            # -------------------------------------------------

            confidence = self._safe_float(
                vlm_result.get(
                    "confidence",
                    0.0,
                ),
                0.0,
            )

            ad_type = (
                vlm_result.get(
                    "ad_type",
                    "other",
                )
                or "other"
            )

            brand = (
                vlm_result.get(
                    "brand",
                    "unknown",
                )
                or "unknown"
            )

            reason = (
                vlm_result.get(
                    "reason",
                    "",
                )
                or ""
            )

            # -------------------------------------------------
            # Source regions
            # -------------------------------------------------

            source_regions = []

            if candidate is not None:

                source_regions = (
                    candidate.get(
                        "source_regions",
                        [],
                    )
                    or []
                )

            # Ensure region indexes are integers.
            clean_source_regions = []

            for region_index in source_regions:

                try:
                    clean_source_regions.append(
                        int(region_index)
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

            source_regions = (
                self._unique_list(
                    clean_source_regions
                )
            )

            # -------------------------------------------------
            # Evidence
            # -------------------------------------------------

            frame_timestamps = []
            frame_paths = []

            for region_index in source_regions:

                region = (
                    evidence_by_region.get(
                        region_index
                    )
                )

                if region is None:
                    continue

                frame = region.get(
                    "frame",
                    {},
                ) or {}

                timestamp = frame.get(
                    "timestamp"
                )

                path = frame.get(
                    "path"
                )

                if timestamp is not None:

                    timestamp_value = (
                        self._safe_float(
                            timestamp,
                            None,
                        )
                    )

                    if timestamp_value is not None:
                        frame_timestamps.append(
                            timestamp_value
                        )

                if path:
                    frame_paths.append(
                        str(path)
                    )

            # -------------------------------------------------
            # Create segment
            # -------------------------------------------------

            advertisement_segments.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "duration": float(
                        end - start
                    ),
                    "ad_type": str(
                        ad_type
                    ),
                    "confidence": float(
                        confidence
                    ),
                    "brand": str(
                        brand
                    ),
                    "description": str(
                        reason
                    ),
                    "source_candidates": (
                        [candidate_index]
                        if candidate_index is not None
                        else []
                    ),
                    "source_regions": (
                        source_regions
                    ),
                    "evidence": {
                        "frame_timestamps": sorted(
                            set(
                                frame_timestamps
                            )
                        ),
                        "frame_paths": (
                            self._unique_list(
                                frame_paths
                            )
                        ),
                    },
                }
            )

        # -----------------------------------------------------
        # Sort by time
        # -----------------------------------------------------

        advertisement_segments.sort(
            key=lambda item: item["start"]
        )

        # -----------------------------------------------------
        # Merge nearby advertisements
        # -----------------------------------------------------

        merged_segments = (
            self._merge_segments(
                advertisement_segments
            )
        )

        # -----------------------------------------------------
        # Assign IDs
        # -----------------------------------------------------

        final_segments = []

        for index, segment in enumerate(
            merged_segments,
            start=1,
        ):

            final_segments.append(
                {
                    "id": (
                        f"long_seg_{index:02d}"
                    ),
                    "start": segment[
                        "start"
                    ],
                    "end": segment[
                        "end"
                    ],
                    "duration": segment[
                        "duration"
                    ],
                    "ad_type": segment[
                        "ad_type"
                    ],
                    "confidence": segment[
                        "confidence"
                    ],
                    "brand": segment[
                        "brand"
                    ],
                    "description": segment[
                        "description"
                    ],
                    "evidence": segment[
                        "evidence"
                    ],
                }
            )

        # -----------------------------------------------------
        # Final output
        # -----------------------------------------------------

        result = {

            "schema_version": "1.0",

            "source": {
                "video": str(
                    video_path
                ),
                "long_vlm_results": str(
                    vlm_path
                ),
                "long_candidates": str(
                    candidates_path
                ),
                "long_evidence": str(
                    evidence_path
                ),
            },

            "method": {
                "merge_gap_seconds": (
                    self.merge_gap_seconds
                ),
                "classification_filter": (
                    "advertisement"
                ),
            },

            "segments": final_segments,

            "stats": {
                "vlm_candidates": len(
                    vlm_results
                ),
                "advertisement_results": len(
                    advertisement_segments
                ),
                "merged_segments": len(
                    final_segments
                ),
            },
        }

        # -----------------------------------------------------
        # Save output
        # -----------------------------------------------------

        output_path = (
            output_dir
            / "long_segments.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                indent=2,
                ensure_ascii=False,
            )

        # -----------------------------------------------------
        # Display
        # -----------------------------------------------------

        print()
        print("=" * 70)
        print(
            "LONG-FORM SEGMENT MERGING"
        )
        print("=" * 70)

        print(
            f"VLM candidates   : "
            f"{len(vlm_results)}"
        )

        print(
            f"Advertisements   : "
            f"{len(advertisement_segments)}"
        )

        print(
            f"Merged segments  : "
            f"{len(final_segments)}"
        )

        print(
            f"Merge gap        : "
            f"{self.merge_gap_seconds:.1f}s"
        )

        print("=" * 70)

        if final_segments:

            print()
            print(
                "FINAL LONG-FORM SEGMENTS"
            )
            print("=" * 70)

            for segment in final_segments:

                print(
                    f"{segment['id']} | "
                    f"{segment['start']:.3f}s -> "
                    f"{segment['end']:.3f}s | "
                    f"duration="
                    f"{segment['duration']:.3f}s | "
                    f"{segment['ad_type']} | "
                    f"{segment['brand']} | "
                    f"confidence="
                    f"{segment['confidence']:.3f}"
                )

        else:

            print()
            print(
                "No advertisements detected."
            )

        print()
        print("=" * 70)

        print(
            f"Results saved: "
            f"{output_path}"
        )

        print("=" * 70)

        return result

    # =========================================================
    # MERGE SEGMENTS
    # =========================================================

    def _merge_segments(
        self,
        segments: list[dict],
    ) -> list[dict]:

        if not segments:
            return []

        merged = [
            dict(segments[0])
        ]

        for current in segments[1:]:

            previous = merged[-1]

            gap = (
                current["start"]
                - previous["end"]
            )

            # Merge overlapping or nearby ads.
            if gap <= self.merge_gap_seconds:

                previous["end"] = max(
                    previous["end"],
                    current["end"],
                )

                previous["duration"] = (
                    previous["end"]
                    - previous["start"]
                )

                # Highest confidence wins.
                previous[
                    "confidence"
                ] = max(
                    previous["confidence"],
                    current["confidence"],
                )

                # Prefer known brand.
                if (
                    previous["brand"]
                    == "unknown"
                    and current["brand"]
                    != "unknown"
                ):

                    previous["brand"] = (
                        current["brand"]
                    )

                # Prefer specific ad type.
                if (
                    previous["ad_type"]
                    == "other"
                    and current["ad_type"]
                    != "other"
                ):

                    previous["ad_type"] = (
                        current["ad_type"]
                    )

                # Prefer non-empty description.
                if (
                    not previous[
                        "description"
                    ]
                    and current[
                        "description"
                    ]
                ):

                    previous[
                        "description"
                    ] = current[
                        "description"
                    ]

                # Merge candidate IDs.
                previous[
                    "source_candidates"
                ] = self._unique_list(
                    previous.get(
                        "source_candidates",
                        [],
                    )
                    + current.get(
                        "source_candidates",
                        [],
                    )
                )

                # Merge source regions.
                previous[
                    "source_regions"
                ] = self._unique_list(
                    previous.get(
                        "source_regions",
                        [],
                    )
                    + current.get(
                        "source_regions",
                        [],
                    )
                )

                # Merge evidence.
                previous_evidence = (
                    previous.get(
                        "evidence",
                        {},
                    )
                )

                current_evidence = (
                    current.get(
                        "evidence",
                        {},
                    )
                )

                previous_evidence[
                    "frame_timestamps"
                ] = sorted(
                    set(
                        previous_evidence.get(
                            "frame_timestamps",
                            [],
                        )
                        + current_evidence.get(
                            "frame_timestamps",
                            [],
                        )
                    )
                )

                previous_evidence[
                    "frame_paths"
                ] = self._unique_list(
                    previous_evidence.get(
                        "frame_paths",
                        [],
                    )
                    + current_evidence.get(
                        "frame_paths",
                        [],
                    )
                )

            else:

                merged.append(
                    dict(current)
                )

        return merged

    # =========================================================
    # JSON LOADER
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

                return json.load(file)

        except FileNotFoundError as error:

            raise LongSegmentMergerError(
                f"File not found: {path}"
            ) from error

        except json.JSONDecodeError as error:

            raise LongSegmentMergerError(
                f"Invalid JSON in {path}: "
                f"{error}"
            ) from error

    # =========================================================
    # SAFE FLOAT
    # =========================================================

    @staticmethod
    def _safe_float(
        value,
        default,
    ):

        if value is None:
            return default

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return default

    # =========================================================
    # UNIQUE LIST
    # =========================================================

    @staticmethod
    def _unique_list(
        values: list,
    ) -> list:

        result = []

        for value in values:

            if value not in result:
                result.append(value)

        return result


# =============================================================
# COMMAND LINE
# =============================================================

def main():

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python -m src.long_segment_merger "
            "<video_path>"
        )

        sys.exit(1)

    video_path = sys.argv[1]

    merger = LongSegmentMerger()

    try:

        merger.merge(
            video_path
        )

    except LongSegmentMergerError as error:

        print()
        print(
            "LONG-FORM SEGMENT MERGING FAILED"
        )

        print(error)

        sys.exit(1)

    except KeyboardInterrupt:

        print()
        print(
            "LONG-FORM SEGMENT MERGING INTERRUPTED"
        )

        sys.exit(130)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    main()