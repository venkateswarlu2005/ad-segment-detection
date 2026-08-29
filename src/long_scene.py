import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2

from src.probe import (
    MediaProbe,
    ProbeError,
)


class LongSceneError(Exception):
    """Raised when long-form scene analysis fails."""


# =============================================================
# DATA STRUCTURES
# =============================================================

@dataclass
class CoarseBoundary:
    timestamp: float
    difference: float


@dataclass
class CoarseRegion:
    region_index: int
    start: float
    end: float
    duration: float
    start_frame: float
    end_frame: float


# =============================================================
# LONG-FORM SCENE DETECTOR
# =============================================================

class LongSceneDetector:
    """
    Coarse scene detector for long-form videos.

    The goal is NOT to detect every scene cut.

    Instead, this stage creates reasonably sized coarse regions
    that can later be examined by OCR / ASR / candidate detection.

    Pipeline:

        video
          |
          v
      sparse sampling
          |
          v
      low-resolution frame comparison
          |
          v
      significant visual changes
          |
          v
      filtered boundaries
          |
          v
      coarse regions
    """

    # ---------------------------------------------------------
    # DEFAULTS
    # ---------------------------------------------------------

    # One frame every 5 seconds.
    DEFAULT_SAMPLE_INTERVAL = 5.0

    # Absolute visual difference threshold.
    DEFAULT_THRESHOLD = 30.0

    # Minimum distance between two accepted boundaries.
    DEFAULT_MIN_BOUNDARY_GAP = 20.0

    # Maximum region duration.
    #
    # This prevents one enormous region from covering a large
    # portion of the video.
    DEFAULT_MAX_REGION_DURATION = 120.0

    # Number of samples used to establish normal visual variation.
    DEFAULT_BASELINE_SAMPLES = 10

    # Relative threshold multiplier.
    #
    # A boundary must be significantly larger than normal
    # background variation.
    DEFAULT_RELATIVE_MULTIPLIER = 1.8

    # ---------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------

    def __init__(
        self,
        sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
        threshold: float = DEFAULT_THRESHOLD,
        min_boundary_gap: float = DEFAULT_MIN_BOUNDARY_GAP,
        max_region_duration: float = DEFAULT_MAX_REGION_DURATION,
        baseline_samples: int = DEFAULT_BASELINE_SAMPLES,
        relative_multiplier: float = DEFAULT_RELATIVE_MULTIPLIER,
    ):
        if sample_interval <= 0:
            raise ValueError(
                "sample_interval must be > 0"
            )

        if threshold < 0:
            raise ValueError(
                "threshold must be >= 0"
            )

        if min_boundary_gap <= 0:
            raise ValueError(
                "min_boundary_gap must be > 0"
            )

        if max_region_duration <= 0:
            raise ValueError(
                "max_region_duration must be > 0"
            )

        if baseline_samples < 1:
            raise ValueError(
                "baseline_samples must be >= 1"
            )

        if relative_multiplier <= 0:
            raise ValueError(
                "relative_multiplier must be > 0"
            )

        self.sample_interval = sample_interval
        self.threshold = threshold
        self.min_boundary_gap = min_boundary_gap
        self.max_region_duration = max_region_duration
        self.baseline_samples = baseline_samples
        self.relative_multiplier = relative_multiplier

    # =========================================================
    # MAIN DETECTION
    # =========================================================

    def detect(
        self,
        video_path: str | Path,
    ) -> dict:

        video_path = Path(video_path)

        if not video_path.exists():
            raise LongSceneError(
                f"Video not found: {video_path}"
            )

        # -----------------------------------------------------
        # Probe metadata
        # -----------------------------------------------------

        try:
            metadata = MediaProbe().probe(
                video_path
            )

        except ProbeError as error:
            raise LongSceneError(
                f"Could not probe video: {error}"
            ) from error

        duration = float(
            metadata.duration
        )

        if duration <= 0:
            raise LongSceneError(
                "Video duration is zero or invalid."
            )

        # -----------------------------------------------------
        # Open video
        # -----------------------------------------------------

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise LongSceneError(
                f"Could not open video: {video_path}"
            )

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        if not fps or fps <= 0:
            fps = 25.0

        # -----------------------------------------------------
        # Display configuration
        # -----------------------------------------------------

        print()
        print("=" * 70)
        print("LONG-FORM COARSE SCENE DETECTION")
        print("=" * 70)

        print(
            f"Video                 : {video_path}"
        )

        print(
            f"Duration              : {duration:.3f}s"
        )

        print(
            f"Sample interval       : "
            f"{self.sample_interval:.1f}s"
        )

        print(
            f"Absolute threshold    : "
            f"{self.threshold:.1f}"
        )

        print(
            f"Minimum boundary gap  : "
            f"{self.min_boundary_gap:.1f}s"
        )

        print(
            f"Maximum region        : "
            f"{self.max_region_duration:.1f}s"
        )

        print(
            f"Baseline samples      : "
            f"{self.baseline_samples}"
        )

        print(
            f"Relative multiplier   : "
            f"{self.relative_multiplier:.2f}"
        )

        print("=" * 70)

        # -----------------------------------------------------
        # Sparse sampling
        # -----------------------------------------------------

        samples = []

        timestamp = 0.0

        while timestamp < duration:

            frame = self._read_frame_at(
                capture,
                timestamp,
                fps,
            )

            if frame is not None:

                samples.append(
                    (
                        timestamp,
                        frame,
                    )
                )

            timestamp += (
                self.sample_interval
            )

        # -----------------------------------------------------
        # Make sure final frame is represented
        # -----------------------------------------------------

        final_timestamp = max(
            0.0,
            duration - 0.05,
        )

        if (
            not samples
            or abs(
                samples[-1][0]
                - final_timestamp
            ) > 0.5
        ):

            frame = self._read_frame_at(
                capture,
                final_timestamp,
                fps,
            )

            if frame is not None:

                samples.append(
                    (
                        final_timestamp,
                        frame,
                    )
                )

        capture.release()

        if not samples:
            raise LongSceneError(
                "No video frames could be sampled."
            )

        print()
        print(
            f"Frames sampled        : "
            f"{len(samples)}"
        )

        # -----------------------------------------------------
        # Calculate frame differences
        # -----------------------------------------------------

        comparisons = []

        previous_frame = None

        for timestamp, frame in samples:

            if previous_frame is None:

                previous_frame = frame
                continue

            difference = (
                self._frame_difference(
                    previous_frame,
                    frame,
                )
            )

            comparisons.append(
                (
                    timestamp,
                    difference,
                )
            )

            previous_frame = frame

        # -----------------------------------------------------
        # Determine baseline
        # -----------------------------------------------------

        baseline = self._calculate_baseline(
            comparisons
        )

        relative_threshold = max(
            self.threshold,
            baseline
            * self.relative_multiplier,
        )

        print()
        print(
            f"Baseline difference   : "
            f"{baseline:.3f}"
        )

        print(
            f"Effective threshold   : "
            f"{relative_threshold:.3f}"
        )

        # -----------------------------------------------------
        # Detect raw boundaries
        # -----------------------------------------------------

        raw_boundaries = []

        for timestamp, difference in comparisons:

            if difference >= relative_threshold:

                raw_boundaries.append(
                    CoarseBoundary(
                        timestamp=round(
                            timestamp,
                            3,
                        ),
                        difference=round(
                            difference,
                            3,
                        ),
                    )
                )

        # -----------------------------------------------------
        # Filter nearby boundaries
        # -----------------------------------------------------

        boundaries = (
            self._filter_boundaries(
                raw_boundaries
            )
        )

        # -----------------------------------------------------
        # Add maximum-duration boundaries
        #
        # This is deliberately separate from visual boundaries.
        # It guarantees that a very long continuous section does
        # not become one massive OCR/VLM candidate region.
        # -----------------------------------------------------

        boundaries = (
            self._add_max_duration_boundaries(
                boundaries,
                duration,
            )
        )

        boundaries.sort(
            key=lambda boundary:
            boundary.timestamp
        )

        # -----------------------------------------------------
        # Build regions
        # -----------------------------------------------------

        regions = self._build_regions(
            duration=duration,
            boundaries=boundaries,
        )

        # -----------------------------------------------------
        # Output
        # -----------------------------------------------------

        output = {
            "schema_version": "1.0",

            "source": {
                "video": str(
                    video_path
                ),
            },

            "method": {
                "sample_interval": (
                    self.sample_interval
                ),

                "difference_threshold": (
                    self.threshold
                ),

                "relative_multiplier": (
                    self.relative_multiplier
                ),

                "effective_threshold": (
                    round(
                        relative_threshold,
                        3,
                    )
                ),

                "minimum_boundary_gap": (
                    self.min_boundary_gap
                ),

                "maximum_region_duration": (
                    self.max_region_duration
                ),

                "baseline_samples": (
                    self.baseline_samples
                ),
            },

            "sampling": {
                "frames_sampled": len(
                    samples
                ),
            },

            "boundaries": [
                asdict(boundary)
                for boundary in boundaries
            ],

            "regions": [
                asdict(region)
                for region in regions
            ],

            "stats": {
                "duration": duration,

                "frames_sampled": len(
                    samples
                ),

                "raw_boundaries": len(
                    raw_boundaries
                ),

                "boundaries": len(
                    boundaries
                ),

                "regions": len(
                    regions
                ),
            },
        }

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        output_path = (
            Path("data")
            / "outputs"
            / video_path.stem
            / "long_scenes.json"
        )

        self._save_json(
            output,
            output_path,
        )

        # -----------------------------------------------------
        # Display
        # -----------------------------------------------------

        self._print_results(
            boundaries,
            regions,
        )

        print()
        print(
            f"Results saved: {output_path}"
        )

        return output

    # =========================================================
    # READ FRAME
    # =========================================================

    @staticmethod
    def _read_frame_at(
        capture,
        timestamp: float,
        fps: float,
    ):

        frame_number = int(
            timestamp * fps
        )

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number,
        )

        success, frame = (
            capture.read()
        )

        if not success:
            return None

        return frame

    # =========================================================
    # FRAME DIFFERENCE
    # =========================================================

    @staticmethod
    def _frame_difference(
        frame_a,
        frame_b,
    ) -> float:

        # Very small representation because this stage only
        # needs coarse visual change detection.

        width = 320
        height = 180

        a = cv2.resize(
            frame_a,
            (width, height),
        )

        b = cv2.resize(
            frame_b,
            (width, height),
        )

        a = cv2.cvtColor(
            a,
            cv2.COLOR_BGR2GRAY,
        )

        b = cv2.cvtColor(
            b,
            cv2.COLOR_BGR2GRAY,
        )

        difference = cv2.absdiff(
            a,
            b,
        )

        return float(
            difference.mean()
        )

    # =========================================================
    # BASELINE
    # =========================================================

    def _calculate_baseline(
        self,
        comparisons,
    ) -> float:

        if not comparisons:
            return 0.0

        values = [
            difference
            for _, difference
            in comparisons
        ]

        # Sort so unusually large scene changes do not distort
        # the estimate of normal variation.

        values.sort()

        count = min(
            self.baseline_samples,
            len(values),
        )

        baseline_values = (
            values[:count]
        )

        if not baseline_values:
            return 0.0

        return float(
            sum(baseline_values)
            / len(baseline_values)
        )

    # =========================================================
    # FILTER BOUNDARIES
    # =========================================================

    def _filter_boundaries(
        self,
        boundaries: list[CoarseBoundary],
    ) -> list[CoarseBoundary]:

        if not boundaries:
            return []

        filtered = []

        for boundary in boundaries:

            if not filtered:

                filtered.append(
                    boundary
                )

                continue

            previous = filtered[-1]

            gap = (
                boundary.timestamp
                - previous.timestamp
            )

            if gap >= self.min_boundary_gap:

                filtered.append(
                    boundary
                )

            else:
                # If several boundaries happen within the same
                # window, retain the strongest one.

                if (
                    boundary.difference
                    > previous.difference
                ):

                    filtered[-1] = (
                        boundary
                    )

        return filtered

    # =========================================================
    # MAXIMUM REGION DURATION
    # =========================================================

    def _add_max_duration_boundaries(
        self,
        boundaries: list[CoarseBoundary],
        duration: float,
    ) -> list[CoarseBoundary]:

        result = list(
            boundaries
        )

        existing_times = {
            round(
                boundary.timestamp,
                3,
            )
            for boundary in result
        }

        # Start from zero and make sure a region cannot exceed
        # the configured maximum duration.

        current = 0.0

        while (
            current
            + self.max_region_duration
            < duration
        ):

            target = (
                current
                + self.max_region_duration
            )

            # Find an existing visual boundary reasonably close
            # to the desired split.

            nearby = None

            for boundary in result:

                distance = abs(
                    boundary.timestamp
                    - target
                )

                if distance <= (
                    self.sample_interval
                ):

                    nearby = boundary
                    break

            if nearby is not None:

                current = (
                    nearby.timestamp
                )

                continue

            # Otherwise create a coarse artificial boundary.

            rounded_target = round(
                target,
                3,
            )

            if (
                rounded_target
                not in existing_times
            ):

                result.append(
                    CoarseBoundary(
                        timestamp=rounded_target,
                        difference=0.0,
                    )
                )

                existing_times.add(
                    rounded_target
                )

            current = target

        return result

    # =========================================================
    # BUILD REGIONS
    # =========================================================

    def _build_regions(
        self,
        duration: float,
        boundaries: list[CoarseBoundary],
    ) -> list[CoarseRegion]:

        boundary_times = sorted(
            {
                boundary.timestamp
                for boundary in boundaries
                if (
                    boundary.timestamp > 0
                    and boundary.timestamp < duration
                )
            }
        )

        points = [
            0.0,
            *boundary_times,
            duration,
        ]

        regions = []

        for index in range(
            len(points) - 1
        ):

            start = points[index]
            end = points[index + 1]

            if end <= start:
                continue

            regions.append(
                CoarseRegion(
                    region_index=(
                        len(regions) + 1
                    ),

                    start=round(
                        start,
                        3,
                    ),

                    end=round(
                        end,
                        3,
                    ),

                    duration=round(
                        end - start,
                        3,
                    ),

                    # Kept for compatibility with
                    # your previous JSON contract.
                    start_frame=round(
                        start,
                        3,
                    ),

                    end_frame=round(
                        end,
                        3,
                    ),
                )
            )

        return regions

    # =========================================================
    # DISPLAY
    # =========================================================

    @staticmethod
    def _print_results(
        boundaries,
        regions,
    ):

        print()
        print("=" * 70)
        print("ACCEPTED COARSE BOUNDARIES")
        print("=" * 70)

        if not boundaries:

            print(
                "No significant visual boundaries detected."
            )

        else:

            for boundary in boundaries:

                boundary_type = (
                    "visual"
                    if boundary.difference > 0
                    else "scheduled"
                )

                print(
                    f"{boundary.timestamp:8.3f}s "
                    f"| difference="
                    f"{boundary.difference:7.3f}"
                    f" | {boundary_type}"
                )

        print()
        print("=" * 70)
        print("COARSE REGIONS")
        print("=" * 70)

        for region in regions:

            print(
                f"Region "
                f"{region.region_index:03d} | "
                f"{region.start:.3f}s -> "
                f"{region.end:.3f}s | "
                f"duration="
                f"{region.duration:.3f}s"
            )

        print("=" * 70)

    # =========================================================
    # SAVE JSON
    # =========================================================

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

def main():

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python -m src.long_scene "
            "<video_path>"
        )

        sys.exit(1)

    video_path = sys.argv[1]

    detector = LongSceneDetector()

    try:

        detector.detect(
            video_path
        )

    except LongSceneError as error:

        print()
        print(
            "LONG-FORM SCENE DETECTION FAILED"
        )

        print(error)

        sys.exit(1)

    except KeyboardInterrupt:

        print()
        print(
            "LONG-FORM SCENE DETECTION INTERRUPTED"
        )

        sys.exit(130)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    main()