import json
import sys
from pathlib import Path

import cv2

from src.asr import ASRProcessor, TranscriptSegment


class LongEvidenceError(Exception):
    """Raised when long-form evidence extraction fails."""


# =============================================================
# CONFIGURATION
# =============================================================

FRAMES_PER_REGION = 1

OUTPUT_ROOT = Path("data") / "outputs"

# Use the existing ASR implementation.
ASR_MODEL = "tiny"


# =============================================================
# LONG-FORM EVIDENCE EXTRACTOR
# =============================================================

class LongEvidenceExtractor:

    def __init__(
        self,
        frames_per_region: int = FRAMES_PER_REGION,
        asr_model: str = ASR_MODEL,
    ):
        self.frames_per_region = max(
            1,
            frames_per_region,
        )

        self.asr_model = asr_model

    # =========================================================
    # MAIN
    # =========================================================

    def extract(
        self,
        video_path: str | Path,
    ) -> dict:

        video_path = Path(video_path)

        if not video_path.exists():
            raise LongEvidenceError(
                f"Video not found: {video_path}"
            )

        output_dir = (
            OUTPUT_ROOT
            / video_path.stem
        )

        scenes_path = (
            output_dir
            / "long_scenes.json"
        )

        if not scenes_path.exists():
            raise LongEvidenceError(
                f"Missing long_scenes.json: "
                f"{scenes_path}"
            )

        # -----------------------------------------------------
        # Load scenes
        # -----------------------------------------------------

        with scenes_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            scenes = json.load(file)

        regions = scenes.get(
            "regions",
            [],
        )

        if not regions:
            raise LongEvidenceError(
                "No regions found in long_scenes.json"
            )

        # -----------------------------------------------------
        # Open video
        # -----------------------------------------------------

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise LongEvidenceError(
                f"Could not open video: "
                f"{video_path}"
            )

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        if not fps or fps <= 0:
            fps = 25.0

        print()
        print("=" * 70)
        print("LONG-FORM EVIDENCE EXTRACTION")
        print("=" * 70)

        print(
            f"Video          : {video_path}"
        )

        print(
            f"Regions        : {len(regions)}"
        )

        print(
            f"Frames/region  : "
            f"{self.frames_per_region}"
        )

        print(
            f"FPS            : {fps:.3f}"
        )

        print("=" * 70)

        # =====================================================
        # ASR
        # =====================================================

        print()
        print("=" * 70)
        print("LONG-FORM AUDIO / ASR")
        print("=" * 70)

        print(
            f"Whisper model  : {self.asr_model}"
        )

        print(
            "Running ASR once for the complete video..."
        )

        try:

            asr = ASRProcessor(
                model_name=self.asr_model
            )

            asr_result = asr.process_video(
                video_path
            )

            transcript_segments = (
                asr_result.get(
                    "segments",
                    []
                )
            )

        except Exception as error:

            capture.release()

            raise LongEvidenceError(
                f"ASR failed: {error}"
            ) from error

        print(
            f"ASR segments   : "
            f"{len(transcript_segments)}"
        )

        print("=" * 70)

        # -----------------------------------------------------
        # Output frame directory
        # -----------------------------------------------------

        evidence = []

        frames_dir = (
            output_dir
            / "long_evidence_frames"
        )

        frames_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =====================================================
        # PROCESS REGIONS
        # =====================================================

        for position, region in enumerate(
            regions,
            start=1,
        ):

            start = float(
                region["start"]
            )

            end = float(
                region["end"]
            )

            duration = max(
                0.0,
                end - start,
            )

            # -------------------------------------------------
            # Middle timestamp
            # -------------------------------------------------

            timestamp = (
                start
                + duration / 2.0
            )

            frame_path = (
                frames_dir
                / f"region_{position:03d}_middle.jpg"
            )

            # -------------------------------------------------
            # Read frame
            # -------------------------------------------------

            frame = self._read_frame(
                capture,
                timestamp,
                fps,
            )

            if frame is None:

                print(
                    f"[{position}/{len(regions)}] "
                    f"FAILED frame at "
                    f"{timestamp:.3f}s"
                )

                continue

            # -------------------------------------------------
            # Save frame
            # -------------------------------------------------

            success = cv2.imwrite(
                str(frame_path),
                frame,
            )

            if not success:

                print(
                    f"[{position}/{len(regions)}] "
                    f"FAILED saving frame"
                )

                continue

            # -------------------------------------------------
            # Transcript for this region
            # -------------------------------------------------

            transcript = self._get_region_transcript(
                transcript_segments,
                start,
                end,
            )

            signals = []

            if transcript:

                signals.append(
                    "speech/transcript detected"
                )

            # -------------------------------------------------
            # Evidence object
            # -------------------------------------------------

            evidence.append(
                {
                    "region_index": (
                        region["region_index"]
                    ),

                    "start": start,

                    "end": end,

                    "duration": duration,

                    "frame": {
                        "timestamp": round(
                            timestamp,
                            3,
                        ),

                        "path": str(
                            frame_path
                        ),
                    },

                    "ocr_text": "",

                    "transcript": transcript,

                    "signals": signals,
                }
            )

            # -------------------------------------------------
            # Console
            # -------------------------------------------------

            transcript_preview = (
                transcript[:100]
                .replace("\n", " ")
                if transcript
                else ""
            )

            print(
                f"[{position}/{len(regions)}] "
                f"Region "
                f"{region['region_index']:03d} | "
                f"{start:.1f}s -> "
                f"{end:.1f}s | "
                f"frame={timestamp:.1f}s | "
                f"speech="
                f"{'yes' if transcript else 'no'}"
            )

            if transcript_preview:

                print(
                    f"  ASR: "
                    f"{transcript_preview}"
                )

        capture.release()

        # =====================================================
        # OUTPUT
        # =====================================================

        result = {

            "schema_version": "1.0",

            "source": {
                "video": str(
                    video_path
                ),

                "long_scenes": str(
                    scenes_path
                ),

                "transcript": str(
                    output_dir
                    / "transcript.json"
                ),
            },

            "method": {

                "frames_per_region": (
                    self.frames_per_region
                ),

                "frame_strategy": (
                    "middle"
                ),

                "asr": {
                    "enabled": True,
                    "model": self.asr_model,
                    "strategy": (
                        "full-video timestamped "
                        "transcript mapped to regions"
                    ),
                },
            },

            "regions": evidence,

            "stats": {

                "total_regions": len(
                    regions
                ),

                "regions_processed": len(
                    evidence
                ),

                "frames_sampled": len(
                    evidence
                ),

                "regions_with_transcript": sum(
                    1
                    for region in evidence
                    if region["transcript"]
                ),

                "asr_segments": len(
                    transcript_segments
                ),
            },
        }

        output_path = (
            output_dir
            / "long_evidence.json"
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

        # =====================================================
        # SUMMARY
        # =====================================================

        regions_with_transcript = sum(
            1
            for region in evidence
            if region["transcript"]
        )

        print()
        print("=" * 70)
        print("LONG-FORM EVIDENCE SUMMARY")
        print("=" * 70)

        print(
            f"Regions processed       : "
            f"{len(evidence)}"
        )

        print(
            f"Frames sampled          : "
            f"{len(evidence)}"
        )

        print(
            f"ASR segments            : "
            f"{len(transcript_segments)}"
        )

        print(
            f"Regions with speech     : "
            f"{regions_with_transcript}"
        )

        print(
            f"Results saved           : "
            f"{output_path}"
        )

        print("=" * 70)

        return result

    # =========================================================
    # REGION TRANSCRIPT
    # =========================================================

    @staticmethod
    def _get_region_transcript(
        segments: list[TranscriptSegment],
        region_start: float,
        region_end: float,
    ) -> str:

        texts = []

        for segment in segments:

            segment_start = float(
                segment.start
            )

            segment_end = float(
                segment.end
            )

            # -------------------------------------------------
            # Overlap test
            #
            # A transcript segment belongs to the region
            # when the two time intervals overlap.
            # -------------------------------------------------

            overlap = (
                segment_start < region_end
                and segment_end > region_start
            )

            if overlap:

                text = str(
                    segment.text
                ).strip()

                if text:

                    texts.append(text)

        return " ".join(texts)

    # =========================================================
    # READ FRAME
    # =========================================================

    @staticmethod
    def _read_frame(
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


# =============================================================
# COMMAND LINE
# =============================================================

def main():

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python -m src.long_evidence "
            "<video_path>"
        )

        sys.exit(1)

    video_path = sys.argv[1]

    extractor = (
        LongEvidenceExtractor()
    )

    try:

        extractor.extract(
            video_path
        )

    except LongEvidenceError as error:

        print()
        print(
            "LONG-FORM EVIDENCE FAILED"
        )

        print(error)

        sys.exit(1)

    except KeyboardInterrupt:

        print()
        print(
            "LONG-FORM EVIDENCE INTERRUPTED"
        )

        sys.exit(130)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    main()