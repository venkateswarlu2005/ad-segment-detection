import base64
import json
import re
import sys
import time
from pathlib import Path

import requests


class LongVLMClassificationError(Exception):
    """Raised when long-form VLM classification fails."""


class LongVLMClassifier:
    """
    VLM verification for long-form commercial candidates.

    Input:
        data/outputs/<video>/long_candidates.json

    Frames:
        data/outputs/<video>/long_evidence_frames/
            region_XXX_middle.jpg

    Output:
        data/outputs/<video>/long_vlm_results.json

    Only candidate regions are sent to Qwen2.5-VL.
    """

    DEFAULT_MODEL = "qwen2.5vl:7b"

    DEFAULT_OLLAMA_URL = (
        "http://localhost:11434/api/chat"
    )

    DEFAULT_TIMEOUT = 900

    DEFAULT_KEEP_ALIVE = "10m"

    DEFAULT_NUM_PREDICT = 180

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = DEFAULT_TIMEOUT,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        num_predict: int = DEFAULT_NUM_PREDICT,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.num_predict = num_predict

        self.session = requests.Session()

    # =========================================================
    # PUBLIC API
    # =========================================================

    def classify_video(
        self,
        video_path: str | Path,
    ) -> dict:

        video_path = Path(video_path)

        if not video_path.exists():
            raise LongVLMClassificationError(
                f"Video not found: {video_path}"
            )

        video_name = video_path.stem

        output_dir = (
            Path("data")
            / "outputs"
            / video_name
        )

        candidates_path = (
            output_dir
            / "long_candidates.json"
        )

        frames_dir = (
            output_dir
            / "long_evidence_frames"
        )

        if not candidates_path.exists():
            raise LongVLMClassificationError(
                f"Candidate file not found: "
                f"{candidates_path}"
            )

        if not frames_dir.exists():
            raise LongVLMClassificationError(
                f"Evidence frames directory not found: "
                f"{frames_dir}"
            )

        data = self._load_json(
            candidates_path
        )

        candidates = data.get("candidate_regions", [])

        if not candidates:
            raise LongVLMClassificationError(
                "No candidates found in "
                "long_candidates.json"
            )

        # -----------------------------------------------------
        # HEADER
        # -----------------------------------------------------

        print()
        print("=" * 70)
        print("LONG-FORM VLM COMMERCIAL VERIFICATION")
        print("=" * 70)

        print(
            f"Model       : {self.model}"
        )

        print(
            f"Video       : {video_path}"
        )

        print(
            f"Candidates  : {len(candidates)}"
        )

        print(
            "Frame mode  : middle representative"
        )

        print(
            "VLM mode    : candidate-only"
        )

        print("=" * 70)

        results = []

        total_time = 0.0

        # -----------------------------------------------------
        # PROCESS CANDIDATES
        # -----------------------------------------------------

        for position, candidate in enumerate(
            candidates,
            start=1,
        ):

            result = self._classify_candidate(
                candidate=candidate,
                frames_dir=frames_dir,
                position=position,
                total=len(candidates),
            )

            results.append(result)

            total_time += result.get(
                "vlm_time_s",
                0.0,
            )

        # -----------------------------------------------------
        # OUTPUT
        # -----------------------------------------------------

        output = self._build_output(
            video_path=video_path,
            candidates=candidates,
            results=results,
            total_time=total_time,
        )

        output_path = (
            output_dir
            / "long_vlm_results.json"
        )

        self._save_json(
            output,
            output_path,
        )

        self._print_summary(
            results=results,
            total_time=total_time,
        )

        print()
        print(
            f"Results saved: {output_path}"
        )

        return output

    # =========================================================
    # CLASSIFY CANDIDATE
    # =========================================================

    def _classify_candidate(
        self,
        candidate: dict,
        frames_dir: Path,
        position: int,
        total: int,
    ) -> dict:

        candidate_index = int(
            candidate.get(
                "candidate_index",
                position,
            )
        )

        start = float(
            candidate["start"]
        )

        end = float(
            candidate["end"]
        )

        source_regions = candidate.get(
            "source_regions",
            [],
        )

        # -----------------------------------------------------
        # Find representative frame
        # -----------------------------------------------------

        frame_path = self._find_frame(
            candidate=candidate,
            frames_dir=frames_dir,
        )

        print()
        print(
            f"[{position}/{total}] "
            f"Candidate {candidate_index:02d} "
            f"| {start:.3f}s -> {end:.3f}s"
        )

        print(
            f"  Source regions : "
            f"{source_regions}"
        )

        print(
            f"  Max score      : "
            f"{float(candidate.get('max_score', 0.0)):.3f}"
        )

        print(
            f"  Image          : "
            f"{frame_path}"
        )

        # -----------------------------------------------------
        # Encode image
        # -----------------------------------------------------

        image_base64 = self._encode_image(
            frame_path
        )

        # -----------------------------------------------------
        # Prompt
        # -----------------------------------------------------

        prompt = self._build_prompt(
            candidate
        )

        # -----------------------------------------------------
        # Ollama request
        # -----------------------------------------------------

        payload = {
            "model": self.model,

            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [
                        image_base64
                    ],
                }
            ],

            "stream": False,

            "keep_alive": self.keep_alive,

            "options": {
                "temperature": 0,
                "num_predict": self.num_predict,
                "num_ctx": 4096,
            },
        }

        start_time = time.perf_counter()

        try:

            response = self.session.post(
                self.ollama_url,
                json=payload,
                timeout=self.timeout,
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            if not response.ok:
                raise LongVLMClassificationError(
                    "Ollama returned "
                    f"HTTP {response.status_code}: "
                    f"{response.text}"
                )

            try:
                response_data = response.json()

            except ValueError as error:

                raise LongVLMClassificationError(
                    "Ollama returned invalid JSON."
                ) from error

            raw_response = (
                response_data
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not raw_response:
                raise LongVLMClassificationError(
                    "VLM returned an empty response."
                )

            parsed = self._parse_json(
                raw_response
            )

            validated = self._validate_result(
                parsed
            )

            validated.update(
                {
                    "candidate_index": candidate_index,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "source_regions": source_regions,
                    "max_score": float(
                        candidate.get(
                            "max_score",
                            0.0,
                        )
                    ),
                    "image": str(
                        frame_path
                    ),
                    "vlm_time_s": round(
                        elapsed,
                        3,
                    ),
                    "vlm_error": None,
                }
            )

            print(
                f"  Classification : "
                f"{validated['classification']}"
            )

            print(
                f"  Confidence     : "
                f"{validated['confidence']:.3f}"
            )

            print(
                f"  Ad type        : "
                f"{validated['ad_type']}"
            )

            print(
                f"  Brand          : "
                f"{validated.get('brand') or 'unknown'}"
            )

            print(
                f"  VLM time       : "
                f"{elapsed:.1f}s"
            )

            return validated

        except Exception as error:

            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"  VLM ERROR      : "
                f"{error}"
            )

            return {
                "classification": "uncertain",
                "confidence": 0.0,
                "ad_type": "other",
                "brand": None,
                "reason": (
                    "VLM classification failed: "
                    f"{error}"
                ),
                "signals": [],
                "candidate_index": candidate_index,
                "start": start,
                "end": end,
                "duration": end - start,
                "source_regions": source_regions,
                "max_score": float(
                    candidate.get(
                        "max_score",
                        0.0,
                    )
                ),
                "image": str(
                    frame_path
                ),
                "vlm_time_s": round(
                    elapsed,
                    3,
                ),
                "vlm_error": str(error),
            }

    # =========================================================
    # FRAME SELECTION
    # =========================================================

    @staticmethod
    def _find_frame(
        candidate: dict,
        frames_dir: Path,
    ) -> Path:

        source_regions = candidate.get(
            "source_regions",
            [],
        )

        # First try the first source region.
        for region_index in source_regions:

            try:
                region_index = int(
                    region_index
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            path = (
                frames_dir
                / (
                    f"region_"
                    f"{region_index:03d}"
                    f"_middle.jpg"
                )
            )

            if path.exists():
                return path

        # Fallback: candidate_index.
        candidate_index = int(
            candidate.get(
                "candidate_index",
                1,
            )
        )

        fallback = (
            frames_dir
            / (
                f"region_"
                f"{candidate_index:03d}"
                f"_middle.jpg"
            )
        )

        if fallback.exists():
            return fallback

        raise LongVLMClassificationError(
            "No representative frame found "
            f"for candidate {candidate_index}"
        )

    # =========================================================
    # PROMPT
    # =========================================================

    @staticmethod
    def _build_prompt(
        candidate: dict,
    ) -> str:

        start = float(
            candidate["start"]
        )

        end = float(
            candidate["end"]
        )

        max_score = float(
            candidate.get(
                "max_score",
                0.0,
            )
        )

        signals = candidate.get(
            "signals",
            [],
        )

        if not signals:
            signals = ["none"]

        return f"""
Classify this long-form video segment as an advertisement or organic content.

You are verifying a candidate segment detected by a commercial-content
detection system.

Segment:
{start:.3f}s -> {end:.3f}s

Candidate score:
{max_score:.3f}

Detector signals:
{", ".join(str(x) for x in signals)}

Use the image as the primary visual evidence.

Advertisement means clear promotional or commercial intent, for example:
- promoting or selling a product
- promoting a business or service
- explicit purchase/order/booking CTA
- price, discount, offer, or contact information used commercially
- sponsor/promotional content
- affiliate promotion

Do NOT classify as an advertisement merely because:
- a person appears
- a doctor appears
- a hospital or business name appears
- a product appears naturally
- a logo or watermark appears
- OCR contains random text
- a phone number appears without promotional context

Be conservative.

Return ONLY valid JSON.

Schema:
{{
  "classification": "advertisement|organic_content|uncertain",
  "confidence": 0.0,
  "ad_type": "product_promotion|service_promotion|self_promo|affiliate|sponsor_read|product_placement|other",
  "brand": "brand name or null",
  "reason": "short evidence-based explanation",
  "signals": ["observable visual or textual signal"]
}}
""".strip()

    # =========================================================
    # IMAGE
    # =========================================================

    @staticmethod
    def _encode_image(
        path: Path,
    ) -> str:

        try:

            return base64.b64encode(
                path.read_bytes()
            ).decode("utf-8")

        except OSError as error:

            raise LongVLMClassificationError(
                f"Could not read image: {path}"
            ) from error

    # =========================================================
    # JSON PARSING
    # =========================================================

    @staticmethod
    def _parse_json(
        text: str,
    ) -> dict:

        text = text.strip()

        # Pure JSON
        try:

            result = json.loads(
                text
            )

            if isinstance(
                result,
                dict,
            ):
                return result

        except json.JSONDecodeError:
            pass

        # Markdown fenced JSON
        cleaned = (
            text
            .replace(
                "```json",
                "",
            )
            .replace(
                "```",
                "",
            )
            .strip()
        )

        try:

            result = json.loads(
                cleaned
            )

            if isinstance(
                result,
                dict,
            ):
                return result

        except json.JSONDecodeError:
            pass

        # First JSON object
        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL,
        )

        if match:

            try:

                result = json.loads(
                    match.group(0)
                )

                if isinstance(
                    result,
                    dict,
                ):
                    return result

            except json.JSONDecodeError:
                pass

        raise LongVLMClassificationError(
            "VLM did not return valid JSON.\n"
            f"Raw response:\n{text}"
        )

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_result(
        result: dict,
    ) -> dict:

        allowed_classifications = {
            "advertisement",
            "organic_content",
            "uncertain",
        }

        allowed_ad_types = {
            "product_promotion",
            "service_promotion",
            "self_promo",
            "affiliate",
            "sponsor_read",
            "product_placement",
            "other",
        }

        classification = result.get(
            "classification"
        )

        if classification not in (
            allowed_classifications
        ):
            raise LongVLMClassificationError(
                "Invalid classification: "
                f"{classification}"
            )

        try:

            confidence = float(
                result.get(
                    "confidence"
                )
            )

        except (
            TypeError,
            ValueError,
        ) as error:

            raise LongVLMClassificationError(
                "Confidence must be numeric."
            ) from error

        if not 0.0 <= confidence <= 1.0:

            raise LongVLMClassificationError(
                "Confidence must be between 0.0 "
                "and 1.0."
            )

        reason = str(
            result.get(
                "reason",
                "",
            )
        ).strip()

        if not reason:
            reason = (
                "No detailed reason returned."
            )

        signals = result.get(
            "signals",
            [],
        )

        if not isinstance(
            signals,
            list,
        ):
            signals = []

        signals = [
            str(signal).strip()
            for signal in signals
            if str(signal).strip()
        ]

        ad_type = result.get(
            "ad_type",
            "other",
        )

        if ad_type not in allowed_ad_types:
            ad_type = "other"

        brand = result.get(
            "brand"
        )

        if brand is not None:

            brand = str(
                brand
            ).strip()

            if not brand:
                brand = None

        return {
            "classification": classification,
            "confidence": round(
                confidence,
                3,
            ),
            "ad_type": ad_type,
            "brand": brand,
            "reason": reason,
            "signals": signals,
        }

    # =========================================================
    # OUTPUT
    # =========================================================

    @staticmethod
    def _build_output(
        video_path: Path,
        candidates: list,
        results: list,
        total_time: float,
    ) -> dict:

        counts = {
            "advertisement": 0,
            "organic_content": 0,
            "uncertain": 0,
        }

        for result in results:

            classification = result.get(
                "classification"
            )

            if classification in counts:

                counts[
                    classification
                ] += 1

        return {
            "schema_version": "1.0",

            "source": {
                "video": str(
                    video_path
                ),
                "long_candidates": str(
                    Path("data")
                    / "outputs"
                    / video_path.stem
                    / "long_candidates.json"
                ),
            },

            "model": {
                "name": "qwen2.5vl:7b",
                "provider": "ollama",
                "temperature": 0,
            },

            "strategy": {
                "candidate_only": True,
                "frame_strategy": "middle",
            },

            "candidates": results,

            "stats": {
                "candidate_count": len(
                    candidates
                ),
                "vlm_calls": len(
                    results
                ),
                "advertisements": counts[
                    "advertisement"
                ],
                "organic_content": counts[
                    "organic_content"
                ],
                "uncertain": counts[
                    "uncertain"
                ],
                "total_vlm_time_s": round(
                    total_time,
                    3,
                ),
            },
        }

    # =========================================================
    # FILE HELPERS
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

        except OSError as error:

            raise LongVLMClassificationError(
                f"Could not read JSON: {path}"
            ) from error

        except json.JSONDecodeError as error:

            raise LongVLMClassificationError(
                f"Invalid JSON: {path}"
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

    # =========================================================
    # SUMMARY
    # =========================================================

    @staticmethod
    def _print_summary(
        results: list,
        total_time: float,
    ) -> None:

        counts = {
            "advertisement": 0,
            "organic_content": 0,
            "uncertain": 0,
        }

        for result in results:

            classification = result.get(
                "classification"
            )

            if classification in counts:

                counts[
                    classification
                ] += 1

        print()
        print("=" * 70)
        print("LONG-FORM VLM SUMMARY")
        print("=" * 70)

        print(
            f"Candidates        : "
            f"{len(results)}"
        )

        print(
            f"Advertisements    : "
            f"{counts['advertisement']}"
        )

        print(
            f"Organic content   : "
            f"{counts['organic_content']}"
        )

        print(
            f"Uncertain         : "
            f"{counts['uncertain']}"
        )

        print(
            f"Total VLM time    : "
            f"{total_time:.1f}s"
        )

        print("=" * 70)


# =============================================================
# COMMAND LINE
# =============================================================

def main():

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python -m src.long_vlm_classifier "
            "<video_path>"
        )

        sys.exit(1)

    video_path = Path(
        sys.argv[1]
    )

    classifier = LongVLMClassifier()

    try:

        classifier.classify_video(
            video_path
        )

    except LongVLMClassificationError as error:

        print()
        print(
            "LONG-FORM VLM CLASSIFICATION FAILED"
        )

        print(error)

        sys.exit(1)

    except KeyboardInterrupt:

        print()
        print(
            "LONG-FORM VLM CLASSIFICATION INTERRUPTED"
        )

        sys.exit(130)


if __name__ == "__main__":
    main()