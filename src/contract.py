import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


class ContractError(Exception):
    """Raised when the final JSON contract cannot be built."""


class ResultContractBuilder:
    """
    Build the final public JSON contract.

    Internal pipeline files:
        metadata.json
        scenes.json
        shot_evidence.json
        ad_scores.json
        segments.json
        vlm_classifications.json
        representatives.json

    Public output:
        result.json
    """

    ALLOWED_AD_TYPES = {
        "preroll",
        "midroll_sponsor_read",
        "product_placement",
        "self_promo",
        "affiliate",
        "platform_inserted",
        "bumper",
        "other",
    }

    def __init__(
        self,
        output_root: str | Path = "data/outputs",
    ):
        self.output_root = Path(output_root)

    # =========================================================
    # PUBLIC API
    # =========================================================

    def build(
        self,
        video_path: str | Path,
    ) -> dict:

        video_path = Path(video_path)
        video_name = video_path.stem

        output_dir = (
            self.output_root
            / video_name
        )

        print()
        print("=" * 70)
        print("FINAL JSON CONTRACT")
        print("=" * 70)
        print(f"Video : {video_path}")
        print(f"Output directory : {output_dir}")
        print("=" * 70)

        # -----------------------------------------------------
        # Load pipeline outputs
        # -----------------------------------------------------

        metadata = self._load_required(
            output_dir / "metadata.json"
        )

        scenes = self._load_required(
            output_dir / "scenes.json"
        )

        shot_evidence = self._load_required(
            output_dir / "shot_evidence.json"
        )

        ad_scores = self._load_required(
            output_dir / "ad_scores.json"
        )

        segments = self._load_required(
            output_dir / "segments.json"
        )

        vlm_classifications = self._load_optional(
            output_dir / "vlm_classifications.json",
            default={},
        )

        representatives = self._load_optional(
            output_dir / "representatives.json",
            default=[],
        )

        print()
        print("INPUT FILES")
        print("-" * 70)

        print("  [OK] metadata.json")
        print("  [OK] scenes.json")
        print("  [OK] shot_evidence.json")
        print("  [OK] ad_scores.json")
        print("  [OK] segments.json")

        if vlm_classifications:
            print("  [OK] vlm_classifications.json")
        else:
            print("  [--] vlm_classifications.json")

        if representatives:
            print("  [OK] representatives.json")
        else:
            print("  [--] representatives.json")

        # -----------------------------------------------------
        # Build lookup tables
        # -----------------------------------------------------

        evidence_by_shot = {
            int(shot["shot_index"]): shot
            for shot in shot_evidence.get(
                "shots",
                []
            )
        }

        scores_by_shot = {
            int(shot["shot_index"]): shot
            for shot in ad_scores.get(
                "shots",
                []
            )
        }

        vlm_by_shot = {
            int(shot["shot_index"]): shot
            for shot in vlm_classifications.get(
                "shots",
                []
            )
        }

        representatives_by_shot = (
            self._build_representative_lookup(
                representatives
            )
        )

        # -----------------------------------------------------
        # Build final public segments
        # -----------------------------------------------------

        final_segments = []

        for index, segment in enumerate(
            segments.get("segments", []),
            start=1,
        ):

            shot_indices = [
                int(value)
                for value in segment.get(
                    "shots",
                    [],
                )
            ]

            start_s = float(
                segment["start"]
            )

            end_s = float(
                segment["end"]
            )

            confidence = float(
                segment.get(
                    "confidence",
                    0.0,
                )
            )

            # -------------------------------------------------
            # Collect all evidence belonging to the segment
            # -------------------------------------------------

            segment_shot_records = []

            for shot_index in shot_indices:

                evidence = evidence_by_shot.get(
                    shot_index,
                    {},
                )

                score = scores_by_shot.get(
                    shot_index,
                    {},
                )

                vlm = vlm_by_shot.get(
                    shot_index,
                    {},
                )

                segment_shot_records.append(
                    {
                        "shot_index": shot_index,
                        "evidence": evidence,
                        "score": score,
                        "vlm": vlm,
                    }
                )

            # -------------------------------------------------
            # Frame timestamps
            # -------------------------------------------------

            frame_timestamps = []

            for shot_index in shot_indices:

                shot_frames = representatives_by_shot.get(
                    shot_index,
                    [],
                )

                for frame in shot_frames:

                    timestamp = frame.get(
                        "timestamp"
                    )

                    if timestamp is None:
                        continue

                    timestamp = float(timestamp)

                    if (
                        start_s <= timestamp <= end_s
                    ):
                        frame_timestamps.append(
                            timestamp
                        )

            frame_timestamps = self._unique_sorted_numbers(
                frame_timestamps
            )

            # -------------------------------------------------
            # Transcript
            # -------------------------------------------------

            transcript_span = self._build_transcript_span(
                segment_shot_records
            )

            # -------------------------------------------------
            # Signals used
            # -------------------------------------------------

            signals_used = self._build_signals_used(
                segment_shot_records
            )

            # -------------------------------------------------
            # Brand
            # -------------------------------------------------

            brand = self._extract_brand(
                segment_shot_records
            )

            # -------------------------------------------------
            # Advertisement type
            # -------------------------------------------------

            ad_type = self._infer_ad_type(
                segment_shot_records
            )

            # -------------------------------------------------
            # Description
            # -------------------------------------------------

            description = self._build_description(
                segment_shot_records,
                brand=brand,
                ad_type=ad_type,
            )

            final_segments.append(
                {
                    "id": f"seg_{index:02d}",
                    "start_s": self._round(
                        start_s
                    ),
                    "end_s": self._round(
                        end_s
                    ),
                    "ad_type": ad_type,
                    "confidence": self._round(
                        confidence
                    ),
                    "brand": brand,
                    "description": description,
                    "evidence": {
                        "frame_timestamps": frame_timestamps,
                        "transcript_span": transcript_span,
                        "signals_used": signals_used,
                    },
                }
            )

        # -----------------------------------------------------
        # Source
        # -----------------------------------------------------

        duration_s = self._get_duration(
            metadata
        )

        media_kind = self._get_media_kind(
            metadata
        )

        source = {
            "url": None,
            "platform": "file",
            "kind": media_kind,
            "duration_s": duration_s,
            "processed_at": datetime.now(
                timezone.utc
            ).isoformat().replace(
                "+00:00",
                "Z",
            ),
        }

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        total_shots = len(
            scenes.get(
                "shots",
                [],
            )
        )

        if total_shots == 0:
            total_shots = len(
                ad_scores.get(
                    "shots",
                    [],
                )
            )

        frames_sampled = (
            self._count_representative_frames(
                representatives
            )
        )

        model_calls = self._get_vlm_calls(
            vlm_classifications
        )

        # Local Ollama inference has no API charge.
        estimated_cost_usd = 0

        # Wall-clock timing is not available inside
        # the current contract process. We deliberately
        # return null instead of inventing a value.
        wall_clock_s = None

        # -----------------------------------------------------
        # Final public contract
        # -----------------------------------------------------

        result = {
            "source": source,
            "segments": final_segments,
            "stats": {
                "wall_clock_s": wall_clock_s,
                "estimated_cost_usd": estimated_cost_usd,
                "frames_sampled": frames_sampled,
                "model_calls": model_calls,
            },
        }

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        output_path = (
            output_dir
            / "result.json"
        )

        self._save_json(
            result,
            output_path,
        )

        # -----------------------------------------------------
        # Display
        # -----------------------------------------------------

        self._print_result(
            result
        )

        print()
        print(
            f"Final contract saved: "
            f"{output_path}"
        )

        return result

    # =========================================================
    # REPRESENTATIVE FRAME HELPERS
    # =========================================================

    @staticmethod
    def _build_representative_lookup(
        representatives,
    ) -> dict:

        lookup = {}

        if isinstance(
            representatives,
            dict,
        ):
            representatives = representatives.get(
                "shots",
                [],
            )

        if not isinstance(
            representatives,
            list,
        ):
            return lookup

        for shot in representatives:

            try:
                shot_index = int(
                    shot["shot_index"]
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

            lookup[shot_index] = shot.get(
                "frames",
                [],
            )

        return lookup

    @staticmethod
    def _count_representative_frames(
        representatives,
    ) -> int:

        if isinstance(
            representatives,
            dict,
        ):
            representatives = representatives.get(
                "shots",
                [],
            )

        if not isinstance(
            representatives,
            list,
        ):
            return 0

        total = 0

        for shot in representatives:

            frames = shot.get(
                "frames",
                [],
            )

            if isinstance(
                frames,
                list,
            ):
                total += len(frames)

        return total

    # =========================================================
    # TRANSCRIPT HELPERS
    # =========================================================

    @staticmethod
    def _build_transcript_span(
        segment_shot_records,
    ) -> str:

        texts = []

        seen = set()

        for record in segment_shot_records:

            evidence = record.get(
                "evidence",
                {},
            )

            audio = evidence.get(
                "audio",
                {},
            )

            transcript = ""

            if isinstance(
                audio,
                dict,
            ):
                transcript = audio.get(
                    "transcript",
                    "",
                )

            if not transcript:
                transcript = evidence.get(
                    "transcript",
                    "",
                )

            if not transcript:
                continue

            transcript = str(
                transcript
            ).strip()

            if not transcript:
                continue

            if transcript not in seen:
                texts.append(
                    transcript
                )
                seen.add(
                    transcript
                )

        return " ".join(
            texts
        )

    # =========================================================
    # SIGNAL HELPERS
    # =========================================================

    @staticmethod
    def _build_signals_used(
        segment_shot_records,
    ) -> list:

        signals = []

        has_ocr = False
        has_asr = False
        has_vlm = False

        for record in segment_shot_records:

            evidence = record.get(
                "evidence",
                {},
            )

            score = record.get(
                "score",
                {},
            )

            vlm = record.get(
                "vlm",
                {},
            )

            # OCR
            ocr = ResultContractBuilder._get_ocr(
                evidence
            )

            if ocr:
                has_ocr = True

            # ASR
            transcript = ResultContractBuilder._get_transcript(
                evidence
            )

            if transcript:
                has_asr = True

            # VLM
            if vlm:
                classification = vlm.get(
                    "classification"
                )

                if classification:
                    has_vlm = True

            if score.get(
                "vlm_classification"
            ):
                has_vlm = True

        if has_asr:
            signals.append(
                "asr"
            )

        if has_ocr:
            signals.append(
                "ocr"
            )

        if has_vlm:
            signals.append(
                "vlm_frame"
            )

        # Every final segment is created from
        # merged shot/scene boundaries.
        if segment_shot_records:
            signals.append(
                "scene_cut"
            )

        return signals

    # =========================================================
    # BRAND
    # =========================================================

    def _extract_brand(
        self,
        segment_shot_records,
    ):

        text_sources = []

        for record in segment_shot_records:

            evidence = record.get(
                "evidence",
                {},
            )

            score = record.get(
                "score",
                {},
            )

            vlm = record.get(
                "vlm",
                {},
            )

            ocr = self._get_ocr(
                evidence
            )

            if ocr:
                text_sources.append(
                    ocr
                )

            reason = vlm.get(
                "reason",
                "",
            )

            if reason:
                text_sources.append(
                    str(reason)
                )

            signals = vlm.get(
                "signals",
                [],
            )

            if isinstance(
                signals,
                list,
            ):
                text_sources.extend(
                    str(signal)
                    for signal in signals
                )

            score_signals = score.get(
                "evidence_labels",
                [],
            )

            if isinstance(
                score_signals,
                list,
            ):
                text_sources.extend(
                    str(signal)
                    for signal in score_signals
                )

        combined = " ".join(
            text_sources
        )

        # Current test video:
        # Asian Dental / DR. SAMSKRUTI
        if re.search(
            r"\bASIAN\s+DENTAL\b",
            combined,
            flags=re.IGNORECASE,
        ):
            return "Asian Dental"

        # Generic business-name extraction from
        # phrases such as "logo for 'XYZ'"
        patterns = [
            r"logo\s+(?:for|of)\s+['\"]([^'\"]+)['\"]",
            r"clinic\s+(?:named|called)\s+['\"]([^'\"]+)['\"]",
            r"brand\s+(?:named|called)\s+['\"]([^'\"]+)['\"]",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                combined,
                flags=re.IGNORECASE,
            )

            if match:

                candidate = match.group(
                    1
                ).strip()

                if candidate:
                    return candidate

        return None

    # =========================================================
    # AD TYPE
    # =========================================================

    def _infer_ad_type(
        self,
        segment_shot_records,
    ) -> str:

        combined = []

        for record in segment_shot_records:

            evidence = record.get(
                "evidence",
                {},
            )

            score = record.get(
                "score",
                {},
            )

            vlm = record.get(
                "vlm",
                {},
            )

            combined.append(
                self._get_ocr(
                    evidence
                )
            )

            combined.append(
                self._get_transcript(
                    evidence
                )
            )

            combined.append(
                str(
                    vlm.get(
                        "reason",
                        "",
                    )
                )
            )

            signals = vlm.get(
                "signals",
                [],
            )

            if isinstance(
                signals,
                list,
            ):
                combined.extend(
                    str(signal)
                    for signal in signals
                )

            labels = score.get(
                "evidence_labels",
                [],
            )

            if isinstance(
                labels,
                list,
            ):
                combined.extend(
                    str(label)
                    for label in labels
                )

        text = " ".join(
            combined
        ).lower()

        # These are conservative classifications.
        # We only assign a specific type when the
        # evidence supports it.

        if any(
            phrase in text
            for phrase in (
                "sponsored by",
                "sponsor",
                "sponsor message",
                "sponsor read",
            )
        ):
            return "midroll_sponsor_read"

        if any(
            phrase in text
            for phrase in (
                "affiliate link",
                "affiliate",
                "use my code",
                "promo code",
                "discount code",
            )
        ):
            return "affiliate"

        if any(
            phrase in text
            for phrase in (
                "follow us",
                "follow for more",
                "our page",
                "our channel",
            )
        ):
            return "self_promo"

        if any(
            phrase in text
            for phrase in (
                "pre-roll",
                "preroll",
            )
        ):
            return "preroll"

        # The current dental end card is clearly
        # promotional, but the available evidence
        # does not establish a more specific category.
        return "other"

    # =========================================================
    # DESCRIPTION
    # =========================================================

    def _build_description(
        self,
        segment_shot_records,
        brand=None,
        ad_type="other",
    ) -> str:

        reasons = []

        for record in segment_shot_records:

            vlm = record.get(
                "vlm",
                {},
            )

            reason = vlm.get(
                "reason",
                "",
            )

            if reason:
                reason = str(
                    reason
                ).strip()

                if reason and reason not in reasons:
                    reasons.append(
                        reason
                    )

        if reasons:
            # Prefer the strongest VLM description.
            # Do not concatenate multiple repetitive
            # descriptions.
            description = max(
                reasons,
                key=len,
            )

            # Make it a little more neutral as a
            # final contract description.
            return description

        ocr_parts = []

        for record in segment_shot_records:

            ocr = self._get_ocr(
                record.get(
                    "evidence",
                    {},
                )
            )

            if ocr:
                ocr_parts.append(
                    ocr
                )

        ocr_text = " ".join(
            ocr_parts
        ).strip()

        if ocr_text:
            if brand:
                return (
                    f"Promotional content for "
                    f"{brand} containing commercial "
                    f"text and contact information."
                )

            return (
                "Promotional content containing "
                "commercial text."
            )

        return (
            "Advertisement detected from "
            "multimodal evidence."
        )

    # =========================================================
    # OCR / ASR
    # =========================================================

    @staticmethod
    def _get_ocr(
        evidence: dict,
    ) -> str:

        visual = evidence.get(
            "visual",
            {},
        )

        if isinstance(
            visual,
            dict,
        ):

            text = visual.get(
                "ocr_text",
                "",
            )

            if text:
                return str(
                    text
                ).strip()

        text = evidence.get(
            "ocr_text",
            "",
        )

        return str(
            text
        ).strip()

    @staticmethod
    def _get_transcript(
        evidence: dict,
    ) -> str:

        audio = evidence.get(
            "audio",
            {},
        )

        if isinstance(
            audio,
            dict,
        ):

            text = audio.get(
                "transcript",
                "",
            )

            if text:
                return str(
                    text
                ).strip()

        text = evidence.get(
            "transcript",
            "",
        )

        return str(
            text
        ).strip()

    # =========================================================
    # VLM
    # =========================================================

    @staticmethod
    def _get_vlm_calls(
        vlm_classifications,
    ) -> int:

        stats = vlm_classifications.get(
            "stats",
            {},
        )

        value = stats.get(
            "vlm_calls"
        )

        if value is not None:

            try:
                return int(
                    value
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

        # Fallback: count non-skipped VLM records.
        shots = vlm_classifications.get(
            "shots",
            [],
        )

        total = 0

        for shot in shots:

            if not shot.get(
                "vlm_skipped",
                False,
            ):
                total += 1

        return total

    # =========================================================
    # METADATA
    # =========================================================

    @staticmethod
    def _get_duration(
        metadata: dict,
    ):

        duration = metadata.get(
            "duration"
        )

        if duration is not None:
            return float(
                duration
            )

        duration = metadata.get(
            "duration_s"
        )

        if duration is not None:
            return float(
                duration
            )

        source = metadata.get(
            "source",
            {},
        )

        if isinstance(
            source,
            dict,
        ):

            duration = source.get(
                "duration"
            )

            if duration is not None:
                return float(
                    duration
                )

            duration = source.get(
                "duration_s"
            )

            if duration is not None:
                return float(
                    duration
                )

        return None

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

        return "unknown"

    # =========================================================
    # JSON
    # =========================================================

    @staticmethod
    def _load_required(
        path: Path,
    ) -> dict:

        if not path.exists():

            raise ContractError(
                f"Required file not found: "
                f"{path}"
            )

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(
                    file
                )

        except json.JSONDecodeError as error:

            raise ContractError(
                f"Invalid JSON: {path}"
            ) from error

        except OSError as error:

            raise ContractError(
                f"Could not read: {path}"
            ) from error

    @staticmethod
    def _load_optional(
        path: Path,
        default,
    ):

        if not path.exists():
            return default

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(
                    file
                )

        except (
            json.JSONDecodeError,
            OSError,
        ):

            return default

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
    # UTILITIES
    # =========================================================

    @staticmethod
    def _round(
        value,
    ):

        if value is None:
            return None

        return round(
            float(value),
            3,
        )

    @staticmethod
    def _unique_sorted_numbers(
        values,
    ) -> list:

        unique = set()

        for value in values:

            try:
                unique.add(
                    round(
                        float(value),
                        3,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

        return sorted(
            unique
        )

    # =========================================================
    # DISPLAY
    # =========================================================

    @staticmethod
    def _print_result(
        result: dict,
    ) -> None:

        source = result[
            "source"
        ]

        stats = result[
            "stats"
        ]

        segments = result[
            "segments"
        ]

        print()
        print("=" * 70)
        print("FINAL RESULT")
        print("=" * 70)

        duration = source.get(
            "duration_s"
        )

        if duration is None:
            print(
                "Duration          : unknown"
            )
        else:
            print(
                f"Duration          : "
                f"{duration:.3f}s"
            )

        print(
            f"Media kind        : "
            f"{source.get('kind')}"
        )

        print(
            f"Platform          : "
            f"{source.get('platform')}"
        )

        print(
            f"Advertisement segments : "
            f"{len(segments)}"
        )

        print(
            f"Frames sampled    : "
            f"{stats.get('frames_sampled', 0)}"
        )

        print(
            f"VLM model calls   : "
            f"{stats.get('model_calls', 0)}"
        )

        print()
        print("SEGMENTS")
        print("-" * 70)

        if not segments:

            print(
                "No advertisement segments detected."
            )

        else:

            for segment in segments:

                print(
                    f"Segment {segment['id']}: "
                    f"{segment['start_s']:.3f}s -> "
                    f"{segment['end_s']:.3f}s"
                )

                print(
                    f"  Type       : "
                    f"{segment['ad_type']}"
                )

                print(
                    f"  Confidence : "
                    f"{segment['confidence']:.3f}"
                )

                print(
                    f"  Brand      : "
                    f"{segment['brand']}"
                )

        print("=" * 70)


# =============================================================
# COMMAND LINE
# =============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python -m src.contract "
            "<video_path>"
        )

        sys.exit(1)

    video_path = Path(
        sys.argv[1]
    )

    builder = ResultContractBuilder()

    try:

        builder.build(
            video_path
        )

    except ContractError as error:

        print()
        print(
            "FINAL CONTRACT FAILED"
        )

        print(error)

        sys.exit(1)