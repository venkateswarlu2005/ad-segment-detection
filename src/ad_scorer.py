import json
import re
import sys
from pathlib import Path


class AdScoringError(Exception):
    """Raised when advertisement scoring fails."""


class AdScorer:
    """
    Deterministic advertisement evidence scorer.

    Combines:

    - VLM classification
    - VLM confidence
    - OCR evidence
    - ASR evidence
    - Commercial CTA signals
    - Contact information
    - Pricing / offer signals

    Output classes:

        advertisement
        organic_content
        uncertain
    """

    # =========================================================
    # SIGNAL DEFINITIONS
    # =========================================================

    STRONG_CTA_PATTERNS = [
        r"\bbook now\b",
        r"\bcall now\b",
        r"\bcontact us\b",
        r"\border now\b",
        r"\bbuy now\b",
        r"\bshop now\b",
        r"\bget yours\b",
        r"\bvisit us\b",
        r"\bsubscribe now\b",
        r"\bdownload now\b",
        r"\bregister now\b",
        r"\bapply now\b",
    ]

    WEAK_CTA_PATTERNS = [
        r"\bfollow us\b",
        r"\bfollow for more\b",
        r"\bfollow for more tips\b",
        r"\blike and follow\b",
        r"\bcheck us out\b",
    ]

    PRICE_PATTERNS = [
        r"₹\s?\d+",
        r"rs\.?\s?\d+",
        r"\b\d+\s?rupees\b",
        r"\$\s?\d+",
        r"€\s?\d+",
        r"£\s?\d+",
        r"\b\d+%\s?(off|discount)\b",
    ]

    OFFER_PATTERNS = [
        r"\bdiscount\b",
        r"\boffer\b",
        r"\bsale\b",
        r"\bdeal\b",
        r"\blimited time\b",
        r"\bspecial price\b",
        r"\bfree\b",
        r"\bflat \d+%\b",
    ]

    CONTACT_PATTERNS = [
        r"\bcontact\b",
        r"\bcall\b",
        r"\bwhatsapp\b",
        r"\bemail\b",
        r"\bwww\.",
        r"\.com\b",
        r"\.in\b",
    ]

    PHONE_PATTERN = re.compile(
        r"(?:\+?\d[\d\s().-]{7,}\d)"
    )

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        advertisement_threshold: float = 0.70,
        uncertain_threshold: float = 0.40,
    ):

        self.advertisement_threshold = (
            advertisement_threshold
        )

        self.uncertain_threshold = (
            uncertain_threshold
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def score_video(
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

        vlm_path = (
            output_dir
            / "vlm_classifications.json"
        )

        evidence_path = (
            output_dir
            / "shot_evidence.json"
        )

        if not vlm_path.exists():
            raise AdScoringError(
                f"VLM results not found: "
                f"{vlm_path}"
            )

        if not evidence_path.exists():
            raise AdScoringError(
                f"Shot evidence not found: "
                f"{evidence_path}"
            )

        vlm_data = self._load_json(
            vlm_path
        )

        evidence_data = self._load_json(
            evidence_path
        )

        vlm_shots = vlm_data.get(
            "shots",
            [],
        )

        evidence_shots = {
            int(
                shot["shot_index"]
            ): shot
            for shot in evidence_data.get(
                "shots",
                [],
            )
        }

        if not vlm_shots:
            raise AdScoringError(
                "No VLM shot results found."
            )

        print()
        print("=" * 70)
        print("ADVERTISEMENT SCORING")
        print("=" * 70)
        print(
            f"Video : {video_path}"
        )
        print(
            f"Shots : {len(vlm_shots)}"
        )
        print(
            f"Advertisement threshold : "
            f"{self.advertisement_threshold:.2f}"
        )
        print(
            f"Uncertain threshold      : "
            f"{self.uncertain_threshold:.2f}"
        )
        print("=" * 70)

        results = []

        for position, vlm_shot in enumerate(
            vlm_shots,
            start=1,
        ):

            shot_index = int(
                vlm_shot["shot_index"]
            )

            evidence = (
                evidence_shots.get(
                    shot_index,
                    {},
                )
            )

            print()
            print(
                f"[{position}/{len(vlm_shots)}] "
                f"Shot {shot_index:02d} "
                f"| "
                f"{vlm_shot['start']:.3f}s → "
                f"{vlm_shot['end']:.3f}s"
            )

            result = self.score_shot(
                vlm_shot=vlm_shot,
                evidence=evidence,
            )

            results.append(
                result
            )

            print(
                f"  Score          : "
                f"{result['ad_score']:.3f}"
            )

            print(
                f"  Classification : "
                f"{result['classification']}"
            )

            print(
                f"  Evidence       : "
                f"{', '.join(result['evidence_labels'])}"
            )

        output = self._build_output(
            video_path=video_path,
            results=results,
        )

        output_path = (
            output_dir
            / "ad_scores.json"
        )

        self._save_json(
            output,
            output_path,
        )

        self._print_summary(
            results
        )

        print()
        print(
            f"Scores saved: {output_path}"
        )

        return output

    # =========================================================
    # SHOT SCORING
    # =========================================================

    def score_shot(
        self,
        vlm_shot: dict,
        evidence: dict,
    ) -> dict:

        ocr_text = self._extract_ocr(
            vlm_shot,
            evidence,
        )

        asr_text = self._extract_asr(
            vlm_shot,
            evidence,
        )

        combined_text = (
            f"{ocr_text} {asr_text}"
        ).lower()

        vlm_classification = (
            vlm_shot.get(
                "classification",
                "uncertain",
            )
        )

        vlm_confidence = float(
            vlm_shot.get(
                "confidence",
                0.0,
            )
        )

        signals = []

        score = 0.0

        # -----------------------------------------------------
        # VLM BASE SIGNAL
        # -----------------------------------------------------

        if vlm_classification == "advertisement":

            score += (
                0.40
                * vlm_confidence
            )

            signals.append(
                "VLM advertisement classification"
            )

        elif vlm_classification == "organic_content":

            score += (
                0.05
                * vlm_confidence
            )

            signals.append(
                "VLM organic classification"
            )

        else:

            signals.append(
                "VLM uncertain classification"
            )

        # -----------------------------------------------------
        # STRONG CTA
        # -----------------------------------------------------

        strong_ctas = self._find_patterns(
            combined_text,
            self.STRONG_CTA_PATTERNS,
        )

        if strong_ctas:

            score += 0.30

            signals.append(
                "strong commercial CTA"
            )

        # -----------------------------------------------------
        # WEAK CTA
        # -----------------------------------------------------

        weak_ctas = self._find_patterns(
            combined_text,
            self.WEAK_CTA_PATTERNS,
        )

        if weak_ctas:

            score += 0.05

            signals.append(
                "social CTA"
            )

        # -----------------------------------------------------
        # PRICE
        # -----------------------------------------------------

        prices = self._find_patterns(
            combined_text,
            self.PRICE_PATTERNS,
        )

        if prices:

            score += 0.15

            signals.append(
                "pricing information"
            )

        # -----------------------------------------------------
        # OFFER
        # -----------------------------------------------------

        offers = self._find_patterns(
            combined_text,
            self.OFFER_PATTERNS,
        )

        if offers:

            score += 0.15

            signals.append(
                "offer/discount language"
            )

        # -----------------------------------------------------
        # CONTACT
        # -----------------------------------------------------

        contacts = self._find_patterns(
            combined_text,
            self.CONTACT_PATTERNS,
        )

        phone_numbers = (
            self.PHONE_PATTERN.findall(
                combined_text
            )
        )

        if contacts or phone_numbers:

            score += 0.15

            signals.append(
                "contact information"
            )

        # -----------------------------------------------------
        # VLM SIGNALS
        # -----------------------------------------------------

        vlm_signals = vlm_shot.get(
            "signals",
            [],
        )

        if isinstance(
            vlm_signals,
            list,
        ):

            commercial_signal_count = (
                self._commercial_signal_count(
                    vlm_signals
                )
            )

            if commercial_signal_count >= 2:

                score += 0.10

                signals.append(
                    "multiple commercial VLM signals"
                )

            elif commercial_signal_count == 1:

                score += 0.05

                signals.append(
                    "commercial VLM signal"
                )

        # -----------------------------------------------------
        # EDUCATIONAL CONTENT PENALTY
        # -----------------------------------------------------

        educational_patterns = [
            r"\btips\b",
            r"\bhow to\b",
            r"\bguide\b",
            r"\beducational\b",
            r"\blearn\b",
            r"\bexplain\b",
            r"\bknowledge\b",
        ]

        educational_signals = (
            self._find_patterns(
                combined_text,
                educational_patterns,
            )
        )

        # Only apply a penalty when there is
        # no strong commercial CTA.

        if (
            educational_signals
            and not strong_ctas
            and not prices
            and not offers
            and not phone_numbers
        ):

            score -= 0.15

            signals.append(
                "educational-content signal"
            )

        # -----------------------------------------------------
        # SOCIAL CTA PENALTY
        # -----------------------------------------------------

        # "Follow us for more" by itself should
        # not make something an advertisement.

        if (
            weak_ctas
            and not strong_ctas
            and not prices
            and not offers
            and not phone_numbers
        ):

            score -= 0.10

            signals.append(
                "social CTA without commercial evidence"
            )

        # -----------------------------------------------------
        # NORMALIZE
        # -----------------------------------------------------

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        # -----------------------------------------------------
        # FINAL CLASSIFICATION
        # -----------------------------------------------------

        if score >= self.advertisement_threshold:

            classification = (
                "advertisement"
            )

        elif score >= self.uncertain_threshold:

            classification = (
                "uncertain"
            )

        else:

            classification = (
                "organic_content"
            )

        return {
            "shot_index": int(
                vlm_shot["shot_index"]
            ),
            "start": float(
                vlm_shot["start"]
            ),
            "end": float(
                vlm_shot["end"]
            ),
            "classification": classification,
            "ad_score": round(
                score,
                3,
            ),
            "vlm_classification": (
                vlm_classification
            ),
            "vlm_confidence": round(
                vlm_confidence,
                3,
            ),
            "evidence_labels": signals,
            "commercial_signals": {
                "strong_cta": strong_ctas,
                "weak_cta": weak_ctas,
                "pricing": prices,
                "offers": offers,
                "contact": contacts,
                "phone_numbers": phone_numbers,
            },
            "ocr_text": ocr_text,
            "transcript": asr_text,
            "vlm_reason": vlm_shot.get(
                "reason",
                "",
            ),
            "vlm_signals": vlm_signals,
        }

    # =========================================================
    # TEXT EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_ocr(
        vlm_shot: dict,
        evidence: dict,
    ) -> str:

        text = vlm_shot.get(
            "ocr_text",
            "",
        )

        if text:
            return str(text)

        return str(
            evidence
            .get("visual", {})
            .get(
                "ocr_text",
                "",
            )
        )

    @staticmethod
    def _extract_asr(
        vlm_shot: dict,
        evidence: dict,
    ) -> str:

        text = vlm_shot.get(
            "transcript",
            "",
        )

        if text:
            return str(text)

        return str(
            evidence
            .get("audio", {})
            .get(
                "transcript",
                "",
            )
        )

    # =========================================================
    # PATTERN HELPERS
    # =========================================================

    @staticmethod
    def _find_patterns(
        text: str,
        patterns: list[str],
    ) -> list[str]:

        found = []

        for pattern in patterns:

            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if matches:

                label = pattern

                found.append(
                    label
                )

        return found

    @staticmethod
    def _commercial_signal_count(
        signals: list,
    ) -> int:

        keywords = [
            "book",
            "buy",
            "contact",
            "phone",
            "price",
            "pricing",
            "product",
            "service",
            "business",
            "clinic",
            "brand",
            "offer",
            "discount",
            "commercial",
            "promotional",
            "call to action",
        ]

        count = 0

        for signal in signals:

            signal_text = str(
                signal
            ).lower()

            if any(
                keyword in signal_text
                for keyword in keywords
            ):

                count += 1

        return count

    # =========================================================
    # OUTPUT
    # =========================================================

    def _build_output(
        self,
        video_path: Path,
        results: list,
    ) -> dict:

        counts = {
            "advertisement": 0,
            "organic_content": 0,
            "uncertain": 0,
        }

        for result in results:

            classification = (
                result["classification"]
            )

            counts[
                classification
            ] += 1

        return {
            "schema_version": "1.0",
            "source": {
                "video": str(
                    video_path
                ),
                "vlm_results": (
                    f"data/outputs/"
                    f"{video_path.stem}/"
                    f"vlm_classifications.json"
                ),
                "shot_evidence": (
                    f"data/outputs/"
                    f"{video_path.stem}/"
                    f"shot_evidence.json"
                ),
            },
            "thresholds": {
                "advertisement": (
                    self.advertisement_threshold
                ),
                "uncertain": (
                    self.uncertain_threshold
                ),
            },
            "shots": results,
            "stats": {
                "total_shots": len(
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
            },
        }

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

            raise AdScoringError(
                f"Invalid JSON: {path}"
            ) from error

        except OSError as error:

            raise AdScoringError(
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

    # =========================================================
    # SUMMARY
    # =========================================================

    @staticmethod
    def _print_summary(
        results: list,
    ) -> None:

        counts = {
            "advertisement": 0,
            "organic_content": 0,
            "uncertain": 0,
        }

        for result in results:

            counts[
                result["classification"]
            ] += 1

        print()
        print("=" * 70)
        print("ADVERTISEMENT SCORING SUMMARY")
        print("=" * 70)

        print(
            f"Total shots       : "
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

        print("=" * 70)


# =============================================================
# COMMAND LINE
# =============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python -m src.ad_scorer "
            "<video_path>"
        )

        sys.exit(1)

    video_path = Path(
        sys.argv[1]
    )

    scorer = AdScorer()

    try:

        scorer.score_video(
            video_path
        )

    except AdScoringError as error:

        print()
        print(
            "ADVERTISEMENT SCORING FAILED"
        )

        print(error)

        sys.exit(1)