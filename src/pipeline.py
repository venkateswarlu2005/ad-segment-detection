import json
import os
import subprocess
import sys
import time
from pathlib import Path


# =============================================================
# UTF-8 CONSOLE SUPPORT
# =============================================================

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


# =============================================================
# ERRORS
# =============================================================

class PipelineError(Exception):
    """Raised when the media pipeline fails."""


# =============================================================
# MEDIA PIPELINE
# =============================================================

class MediaPipeline:
    """
    End-to-end media processing orchestrator.

    SHORT-FORM:

        Probe
            -> Scene Detection
            -> Representative Frames
            -> Shot OCR
            -> ASR
            -> Shot ASR Alignment
            -> Shot Evidence
            -> VLM Classification
            -> Ad Scoring
            -> Segment Merging
            -> Final Contract

    LONG-FORM:

        Probe
            -> ASR
            -> Long Scene Detection
            -> Long Evidence
            -> Long Candidate Detection
            -> Long VLM Classification
            -> Long Segment Merging
            -> Final Contract
    """

    def __init__(
        self,
        python_executable: str | None = None,
    ):
        self.python_executable = (
            python_executable
            or sys.executable
        )

        self.module_timings: dict[str, float] = {}
        self.pipeline_start_time: float | None = None

    # =========================================================
    # PUBLIC API
    # =========================================================

    def process(
        self,
        video_path: str | Path,
    ) -> dict:

        self.pipeline_start_time = time.perf_counter()
        self.module_timings = {}

        video_path = Path(video_path)

        if not video_path.exists():
            raise PipelineError(
                f"Video does not exist: {video_path}"
            )

        if not video_path.is_file():
            raise PipelineError(
                f"Video is not a file: {video_path}"
            )

        print()
        print("=" * 70)
        print("MEDIA PIPELINE")
        print("=" * 70)
        print(f"Video : {video_path}")
        print("=" * 70)

        # -----------------------------------------------------
        # STEP 1: PROBE
        # -----------------------------------------------------

        self._run_module(
            "src.probe",
            video_path,
        )

        output_dir = (
            Path("data")
            / "outputs"
            / video_path.stem
        )

        metadata_path = (
            output_dir
            / "metadata.json"
        )

        metadata = self._load_json(
            metadata_path
        )

        media_kind = self._get_media_kind(
            metadata
        )

        print()
        print("=" * 70)
        print("ROUTING")
        print("=" * 70)
        print(f"Media kind : {media_kind}")

        # -----------------------------------------------------
        # ROUTE
        # -----------------------------------------------------

        if media_kind == "short":

            print("Pipeline   : short_form")
            print("=" * 70)

            return self._run_short_pipeline(
                video_path
            )

        if media_kind == "long_form":

            print("Pipeline   : long_form")
            print("=" * 70)

            return self._run_long_pipeline(
                video_path
            )

        raise PipelineError(
            f"Unsupported media kind: {media_kind}"
        )

    # =========================================================
    # SHORT FORM
    # =========================================================

    def _run_short_pipeline(
        self,
        video_path: Path,
    ) -> dict:

        print()
        print("=" * 70)
        print("SHORT-FORM PIPELINE")
        print("=" * 70)

        # -----------------------------------------------------
        # 1. Scene detection
        # -----------------------------------------------------

        print()
        print("[SHORT] Step 1/10 - Scene detection")

        self._run_module(
            "src.scene",
            video_path,
        )

        # -----------------------------------------------------
        # 2. Representative frames
        # -----------------------------------------------------

        print()
        print("[SHORT] Step 2/10 - Representative frames")

        self._run_module(
            "src.shots",
            video_path,
        )

        # -----------------------------------------------------
        # 3. OCR
        # -----------------------------------------------------

        print()
        print("[SHORT] Step 3/10 - Shot OCR")

        self._run_module(
            "src.shot_ocr",
            video_path,
        )

        # -----------------------------------------------------
        # 4. ASR
        # -----------------------------------------------------

        print()
        print("[SHORT] Step 4/10 - Audio + ASR")

        self._run_module(
            "src.asr",
            video_path,
        )

        # -----------------------------------------------------
        # VERIFY ASR OUTPUT
        # -----------------------------------------------------

        transcript_path = self._verify_transcript(
            video_path
        )

        print(
            f"[SHORT] ASR verified: {transcript_path}"
        )

        # -----------------------------------------------------
        # 5. ASR -> shot alignment
        # -----------------------------------------------------

        print()
        print("[SHORT] Step 5/10 - ASR → shot alignment")

        self._run_module(
            "src.shot_asr",
            video_path,
        )

        # Verify shot ASR output

        shot_asr_path = (
            Path("data")
            / "outputs"
            / video_path.stem
            / "shot_asr.json"
        )

        self._verify_output(
            shot_asr_path,
            "Shot ASR"
        )

        # -----------------------------------------------------
        # 6. Combine OCR + ASR evidence
        # -----------------------------------------------------

        print()
        print("[SHORT] Step 6/10 - Shot evidence")

        self._run_module(
            "src.shot_evidence",
            video_path,
        )

        # -----------------------------------------------------
        # 7. VLM classification
        # -----------------------------------------------------

        print()
        print("[SHORT] Step 7/10 - VLM classification")

        self._run_module(
            "src.vlm_classifier",
            video_path,
        )

        # -----------------------------------------------------
        # 8. Advertisement scoring
        # -----------------------------------------------------

        print()
        print("[SHORT] Step 8/10 - Advertisement scoring")

        self._run_module(
            "src.ad_scorer",
            video_path,
        )

        # -----------------------------------------------------
        # 9. Segment merging
        # -----------------------------------------------------

        print()
        print("[SHORT] Step 9/10 - Segment merging")

        self._run_module(
            "src.segment_merger",
            video_path,
        )

        # -----------------------------------------------------
        # 10. Final JSON contract
        # -----------------------------------------------------

        print()
        print("[SHORT] Step 10/10 - Final contract")

        self._run_module(
            "src.contract",
            video_path,
        )

        # -----------------------------------------------------
        # Load final result
        # -----------------------------------------------------

        result_path = (
            Path("data/outputs")
            / video_path.stem
            / "result.json"
        )

        result = self._load_json(
            result_path
        )

        self._finalize_result(
            result_path,
            result,
        )

        self._print_completion(
            video_path,
            result,
        )

        return result

    # =========================================================
    # LONG FORM
    # =========================================================

    def _run_long_pipeline(
        self,
        video_path: Path,
    ) -> dict:

        print()
        print("=" * 70)
        print("LONG-FORM PIPELINE")
        print("=" * 70)

        # -----------------------------------------------------
        # 1. ASR
        # -----------------------------------------------------

        print()
        print("[LONG] Step 1/7 - ASR")

        self._run_module(
            "src.asr",
            video_path,
        )

        # -----------------------------------------------------
        # VERIFY ASR OUTPUT
        # -----------------------------------------------------

        transcript_path = self._verify_transcript(
            video_path
        )

        print(
            f"[LONG] ASR verified: {transcript_path}"
        )

        # -----------------------------------------------------
        # 2. LONG SCENE DETECTION
        # -----------------------------------------------------

        print()
        print("[LONG] Step 2/7 - Long scene detection")

        self._run_module(
            "src.long_scene",
            video_path,
        )

        # -----------------------------------------------------
        # 3. LONG EVIDENCE
        # -----------------------------------------------------

        print()
        print("[LONG] Step 3/7 - Long evidence extraction")

        self._run_module(
            "src.long_evidence",
            video_path,
        )

        # -----------------------------------------------------
        # 4. LONG CANDIDATE DETECTION
        # -----------------------------------------------------

        print()
        print("[LONG] Step 4/7 - Candidate detection")

        self._run_module(
            "src.long_candidate_detector",
            video_path,
        )

        # -----------------------------------------------------
        # 5. LONG VLM CLASSIFICATION
        # -----------------------------------------------------

        print()
        print("[LONG] Step 5/7 - VLM commercial verification")

        self._run_module(
            "src.long_vlm_classifier",
            video_path,
        )

        # -----------------------------------------------------
        # 6. LONG SEGMENT MERGING
        # -----------------------------------------------------

        print()
        print("[LONG] Step 6/7 - Segment merging")

        self._run_module(
            "src.long_segment_merger",
            video_path,
        )

        # -----------------------------------------------------
        # 7. FINAL CONTRACT
        # -----------------------------------------------------

        print()
        print("[LONG] Step 7/7 - Final result contract")

        result = self._build_long_final_result(
            video_path
        )

        result_path = (
            Path("data")
            / "outputs"
            / video_path.stem
            / "result.json"
        )

        self._save_json(
            result_path,
            result,
        )

        self._print_completion(
            video_path,
            result,
        )

        return result

    # =========================================================
    # VERIFY TRANSCRIPT
    # =========================================================

    def _verify_transcript(
        self,
        video_path: Path,
    ) -> Path:

        transcript_path = (
            Path("data")
            / "outputs"
            / video_path.stem
            / "transcript.json"
        )

        if not transcript_path.exists():
            raise PipelineError(
                "ASR completed but transcript.json was not created.\n"
                f"Expected: {transcript_path}\n\n"
                "The ASR module must create transcript.json "
                "for the requested video before shot_asr runs."
            )

        if not transcript_path.is_file():
            raise PipelineError(
                f"Transcript path is not a file: "
                f"{transcript_path}"
            )

        try:

            with transcript_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

        except json.JSONDecodeError as error:

            raise PipelineError(
                f"ASR created invalid JSON: "
                f"{transcript_path}"
            ) from error

        except OSError as error:

            raise PipelineError(
                f"Could not read ASR transcript: "
                f"{transcript_path}: {error}"
            ) from error

        if not isinstance(data, dict):
            raise PipelineError(
                f"Invalid transcript structure: "
                f"{transcript_path}"
            )

        return transcript_path

    # =========================================================
    # VERIFY OUTPUT
    # =========================================================

    @staticmethod
    def _verify_output(
        path: Path,
        stage_name: str,
    ) -> Path:

        if not path.exists():

            raise PipelineError(
                f"{stage_name} completed but expected "
                f"output was not created:\n"
                f"{path}"
            )

        if not path.is_file():

            raise PipelineError(
                f"{stage_name} output is not a file:\n"
                f"{path}"
            )

        return path

    # =========================================================
    # BUILD LONG-FORM FINAL RESULT
    # =========================================================

    def _build_long_final_result(
        self,
        video_path: Path,
    ) -> dict:

        output_dir = (
            Path("data")
            / "outputs"
            / video_path.stem
        )

        metadata_path = (
            output_dir
            / "metadata.json"
        )

        long_segments_path = (
            output_dir
            / "long_segments.json"
        )

        vlm_results_path = (
            output_dir
            / "long_vlm_results.json"
        )

        candidates_path = (
            output_dir
            / "long_candidates.json"
        )

        metadata = self._load_json(
            metadata_path
        )

        long_segments = self._load_json(
            long_segments_path
        )

        # -----------------------------------------------------
        # Source information
        # -----------------------------------------------------

        source_metadata = (
            long_segments.get(
                "source",
                {}
            )
        )

        duration = (
            metadata.get("duration")
            or metadata.get("duration_s")
            or source_metadata.get("duration")
            or source_metadata.get("duration_s")
        )

        media_kind = (
            metadata.get("media_kind")
            or metadata.get("kind")
            or "long_form"
        )

        # -----------------------------------------------------
        # Locate segments
        # -----------------------------------------------------

        segments = long_segments.get(
            "segments",
            []
        )

        if not segments:

            segments = long_segments.get(
                "long_segments",
                []
            )

        final_segments = []

        for index, segment in enumerate(
            segments,
            start=1,
        ):

            start = float(
                segment.get(
                    "start_s",
                    segment.get(
                        "start",
                        0.0,
                    ),
                )
            )

            end = float(
                segment.get(
                    "end_s",
                    segment.get(
                        "end",
                        start,
                    ),
                )
            )

            duration_s = float(
                segment.get(
                    "duration_s",
                    segment.get(
                        "duration",
                        max(
                            0.0,
                            end - start,
                        ),
                    ),
                )
            )

            confidence = float(
                segment.get(
                    "confidence",
                    0.0,
                )
            )

            final_segments.append(
                {
                    "segment_id": segment.get(
                        "segment_id",
                        segment.get(
                            "id",
                            f"long_seg_{index:02d}",
                        ),
                    ),
                    "start_s": start,
                    "end_s": end,
                    "duration_s": duration_s,
                    "classification": "advertisement",
                    "ad_type": segment.get(
                        "ad_type",
                        "other",
                    ),
                    "brand": segment.get(
                        "brand"
                    ),
                    "confidence": confidence,
                    "source": "long_form",
                }
            )

        # -----------------------------------------------------
        # Advertisement duration
        # -----------------------------------------------------

        advertisement_duration = sum(
            segment["duration_s"]
            for segment in final_segments
        )

        # -----------------------------------------------------
        # Candidate / VLM statistics
        # -----------------------------------------------------

        candidate_count = 0
        vlm_calls = 0
        advertisements = 0
        organic_content = 0
        uncertain = 0
        vlm_time = 0.0

        if candidates_path.exists():

            try:

                candidates_data = (
                    self._load_json(
                        candidates_path
                    )
                )

                candidate_count = len(
                    candidates_data.get(
                        "candidates",
                        []
                    )
                )

            except PipelineError:

                candidate_count = 0

        if vlm_results_path.exists():

            try:

                vlm_data = (
                    self._load_json(
                        vlm_results_path
                    )
                )

                vlm_candidates = (
                    vlm_data.get(
                        "candidates",
                        []
                    )
                )

                vlm_stats = (
                    vlm_data.get(
                        "stats",
                        {}
                    )
                )

                vlm_calls = int(
                    vlm_stats.get(
                        "vlm_calls",
                        len(vlm_candidates),
                    )
                )

                advertisements = int(
                    vlm_stats.get(
                        "advertisements",
                        0,
                    )
                )

                organic_content = int(
                    vlm_stats.get(
                        "organic_content",
                        0,
                    )
                )

                uncertain = int(
                    vlm_stats.get(
                        "uncertain",
                        0,
                    )
                )

                vlm_time = float(
                    vlm_stats.get(
                        "total_vlm_time_s",
                        0.0,
                    )
                )

            except PipelineError:
                pass

        # -----------------------------------------------------
        # Wall-clock runtime
        # -----------------------------------------------------

        wall_clock = (
            time.perf_counter()
            - self.pipeline_start_time
            if self.pipeline_start_time is not None
            else None
        )

        # -----------------------------------------------------
        # Final contract
        # -----------------------------------------------------

        result = {
            "schema_version": "1.0",

            "source": {
                "video": str(
                    video_path
                ),
                "duration_s": (
                    float(duration)
                    if duration is not None
                    else None
                ),
                "kind": str(
                    media_kind
                ),
            },

            "pipeline": {
                "media_kind": "long_form",
                "status": "complete",
                "architecture": [
                    "probe",
                    "asr",
                    "long_scene",
                    "long_evidence",
                    "long_candidate_detector",
                    "long_vlm_classifier",
                    "long_segment_merger",
                ],
            },

            "segments": final_segments,

            "stats": {
                "detected_segments": len(
                    final_segments
                ),

                "advertisement_duration": round(
                    advertisement_duration,
                    3,
                ),

                "candidate_count": (
                    candidate_count
                ),

                "model_calls": (
                    vlm_calls
                ),

                "advertisements_verified": (
                    advertisements
                ),

                "organic_candidates": (
                    organic_content
                ),

                "uncertain_candidates": (
                    uncertain
                ),

                "vlm_time_s": round(
                    vlm_time,
                    3,
                ),

                "wall_clock_s": (
                    round(
                        wall_clock,
                        3,
                    )
                    if wall_clock is not None
                    else None
                ),

                "module_timings_s": {
                    name: round(
                        duration,
                        3,
                    )
                    for name, duration
                    in self.module_timings.items()
                },
            },
        }

        return result

    # =========================================================
    # RUN MODULE
    # =========================================================

    def _run_module(
        self,
        module_name: str,
        video_path: Path,
    ) -> None:

        print()
        print("=" * 70)
        print(
            f"[PIPELINE] Running {module_name}"
        )
        print("=" * 70)

        command = [
            self.python_executable,
            "-m",
            module_name,
            str(video_path),
        ]

        # -----------------------------------------------------
        # CHILD PROCESS ENVIRONMENT
        # -----------------------------------------------------

        child_env = os.environ.copy()

        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"

        start_time = time.perf_counter()

        try:

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                universal_newlines=True,
                env=child_env,
            )

        except OSError as error:

            raise PipelineError(
                f"Could not start {module_name}: "
                f"{error}"
            ) from error

        # -----------------------------------------------------
        # STREAM CHILD OUTPUT
        # -----------------------------------------------------

        try:

            if process.stdout is not None:

                for line in process.stdout:

                    print(
                        line.rstrip("\r\n"),
                        flush=True,
                    )

        except KeyboardInterrupt:

            print()
            print(
                f"[PIPELINE] Stopping {module_name}..."
            )

            process.terminate()

            try:

                process.wait(
                    timeout=5
                )

            except subprocess.TimeoutExpired:

                process.kill()

            raise PipelineError(
                f"Pipeline interrupted while "
                f"running {module_name}"
            )

        finally:

            if process.stdout is not None:
                process.stdout.close()

        return_code = process.wait()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        self.module_timings[
            module_name
        ] = elapsed

        # -----------------------------------------------------
        # HANDLE FAILURE
        # -----------------------------------------------------

        if return_code != 0:

            raise PipelineError(
                f"Module failed: "
                f"{module_name} "
                f"(exit code {return_code})"
            )

        print()
        print(
            f"[PIPELINE] {module_name} "
            f"completed in "
            f"{elapsed:.2f}s."
        )

    # =========================================================
    # METADATA
    # =========================================================

    @staticmethod
    def _get_media_kind(
        metadata: dict,
    ) -> str:

        media_kind = metadata.get(
            "media_kind"
        )

        if media_kind:

            return str(
                media_kind
            )

        media_kind = metadata.get(
            "kind"
        )

        if media_kind:

            return str(
                media_kind
            )

        raise PipelineError(
            "metadata.json does not contain "
            "'media_kind'."
        )

    # =========================================================
    # JSON LOAD
    # =========================================================

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict:

        if not path.exists():

            raise PipelineError(
                f"Expected output not found: "
                f"{path}"
            )

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

        except json.JSONDecodeError as error:

            raise PipelineError(
                f"Invalid JSON: {path}"
            ) from error

        except OSError as error:

            raise PipelineError(
                f"Could not read: {path}"
            ) from error

        if not isinstance(
            data,
            dict,
        ):

            raise PipelineError(
                f"Expected JSON object: "
                f"{path}"
            )

        return data

    # =========================================================
    # JSON SAVE
    # =========================================================

    @staticmethod
    def _save_json(
        path: Path,
        data: dict,
    ) -> None:

        try:

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

        except OSError as error:

            raise PipelineError(
                f"Could not write: {path}"
            ) from error

    # =========================================================
    # FINALIZE RESULT
    # =========================================================

    def _finalize_result(
        self,
        result_path: Path,
        result: dict,
    ) -> None:

        total_runtime = (
            time.perf_counter()
            - self.pipeline_start_time
            if self.pipeline_start_time is not None
            else None
        )

        if total_runtime is None:
            return

        stats = result.setdefault(
            "stats",
            {}
        )

        stats["wall_clock_s"] = round(
            total_runtime,
            3,
        )

        stats["module_timings_s"] = {
            name: round(
                duration,
                3,
            )
            for name, duration
            in self.module_timings.items()
        }

        self._save_json(
            result_path,
            result,
        )

    # =========================================================
    # COMPLETION
    # =========================================================

    def _print_completion(
        self,
        video_path: Path,
        result: dict,
    ) -> None:

        source = result.get(
            "source",
            {},
        )

        stats = result.get(
            "stats",
            {},
        )

        segments = result.get(
            "segments",
            [],
        )

        print()
        print("=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)

        print(
            f"Video     : {video_path}"
        )

        print(
            f"Duration  : "
            f"{source.get('duration_s', 'unknown')}s"
        )

        print(
            f"Media kind: "
            f"{source.get('kind', 'unknown')}"
        )

        print(
            f"Segments  : "
            f"{len(segments)}"
        )

        advertisement_duration = stats.get(
            "advertisement_duration",
            0,
        )

        print(
            f"Ad time   : "
            f"{advertisement_duration}s"
        )

        wall_clock = stats.get(
            "wall_clock_s"
        )

        if wall_clock is not None:

            print(
                f"Runtime   : "
                f"{wall_clock}s"
            )

        model_calls = stats.get(
            "model_calls"
        )

        if model_calls is not None:

            print(
                f"Model calls: "
                f"{model_calls}"
            )

        candidate_count = stats.get(
            "candidate_count"
        )

        if candidate_count is not None:

            print(
                f"Candidates : "
                f"{candidate_count}"
            )

        print()
        print("MODULE TIMINGS")
        print("-" * 70)

        if self.module_timings:

            for (
                module_name,
                duration,
            ) in self.module_timings.items():

                print(
                    f"{module_name:<35} "
                    f"{duration:>10.2f}s"
                )

        else:

            print(
                "No module timing data."
            )

        print()
        print("DETECTED SEGMENTS")
        print("-" * 70)

        if not segments:

            print(
                "No advertisement segments detected."
            )

        else:

            for index, segment in enumerate(
                segments,
                start=1,
            ):

                start = float(
                    segment.get(
                        "start_s",
                        segment.get(
                            "start",
                            0,
                        ),
                    )
                )

                end = float(
                    segment.get(
                        "end_s",
                        segment.get(
                            "end",
                            0,
                        ),
                    )
                )

                duration = float(
                    segment.get(
                        "duration_s",
                        segment.get(
                            "duration",
                            end - start,
                        ),
                    )
                )

                confidence = float(
                    segment.get(
                        "confidence",
                        0,
                    )
                )

                ad_type = segment.get(
                    "ad_type",
                    "other",
                )

                brand = segment.get(
                    "brand"
                )

                print(
                    f"Segment {index:02d}: "
                    f"{start:.3f}s -> "
                    f"{end:.3f}s "
                    f"({duration:.3f}s) "
                    f"| type={ad_type} "
                    f"| brand={brand} "
                    f"| confidence="
                    f"{confidence:.3f}"
                )

        print("=" * 70)

        print("Final result:")

        print(
            Path("data/outputs")
            / video_path.stem
            / "result.json"
        )


# =============================================================
# COMMAND LINE
# =============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python -m src.pipeline "
            "<video_path>"
        )

        sys.exit(1)

    video_path = Path(
        sys.argv[1]
    )

    pipeline = MediaPipeline()

    try:

        pipeline.process(
            video_path
        )

    except PipelineError as error:

        print()
        print("=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)
        print(error)

        sys.exit(1)

    except KeyboardInterrupt:

        print()
        print("=" * 70)
        print("PIPELINE INTERRUPTED")
        print("=" * 70)

        sys.exit(130)