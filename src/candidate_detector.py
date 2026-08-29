
import json
import re
import sys
from pathlib import Path


# =============================================================
# UTF-8 SUPPORT
# =============================================================

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


class CandidateDetectionError(Exception):
    """Raised when candidate detection fails."""


class CandidateDetector:
    """
    Lightweight commercial-candidate detector.

    This module DOES NOT call the VLM.

    It examines:
        - OCR evidence
        - ASR evidence
        - commercial keywords
        - CTA language
        - phone numbers
        - prices
        - business/contact information
        - neighboring shots

    Its job is candidate generation, not final advertisement
    classification.

    Output:
        data/outputs/<video>/candidates.json
    """

    # =========================================================
    # COMMERCIAL SIGNALS
    # =========================================================

    STRONG_KEYWORDS = {
        "book now",
        "buy now",
        "shop now",
        "order now",
        "contact us",
        "call now",
        "visit us",
        "learn more",
        "click here",
        "sign up",
        "get yours",
        "try now",
        "download now",
        "subscribe",
        "follow us",
        "follow for more",
        "dm us",
        "message us",
    }

    COMMERCIAL_KEYWORDS = {
        "offer",
        "discount",
        "sale",
        "deal",
        "special offer",
        "limited time",
        "price",
        "prices",
        "cost",
        "costs",
        "starting from",
        "available",
        "appointment",
        "booking",
        "delivery",
        "purchase",
        "order",
        "product",
        "products",
        "service",
        "services",
        "clinic",
        "hospital",
        "doctor",
        "dentist",
        "dental",
        "restaurant",
        "store",
        "shop",
        "brand",
        "website",
        "whatsapp",
        "instagram",
        "youtube",
    }

    # Phrases that can indicate promotional/product advice even
    # without an explicit CTA.
    PROMOTIONAL_PHRASES = {
        "pea-sized amount",
        "pea sized amount",
        "recommended amount",
        "recommended dosage",
        "how to use",
        "use this",
        "use our",
        "our product",
        "our service",
        "best product",
        "best service",
        "for more tips",
        "for more dental care tips",
        "tips",
    }

    BUSINESS_TERMS = {
        "clinic",
        "hospital",
        "doctor",
        "dr.",
        "dentist",
        "dental",
        "medical",
        "appointment",
        "address",
        "road no",
        "road number",
        "hyderabad",
        "jubilee hills",
        "madhapur",
        "contact",
        "phone",
        "mobile",
        "email",
    }

    # =========================================================
    # INIT
    # =========================================================

    def __init__(
        self,
        candidate_threshold: float = 0.20,
        bridge_threshold: float = 0.10,
        max_bridge_gap: float = 1.50,
    ):
        self.candidate_threshold = candidate_threshold
        self.bridge_threshold = bridge_threshold
        self.max_bridge_gap = max_bridge_gap

    # =========================================================
    # PUBLIC API
    # =========================================================

    def detect(
        self,
        video_path: str | Path,
    ) -> dict:

        video_path = Path(video_path)
        video_name = video_path.stem

        output_dir = (
            Path("data/outputs")
            / video_name
        )

        evidence_path = (
            output_dir
            / "shot_evidence.json"
        )

        if not evidence_path.exists():
            raise CandidateDetectionError(
                f"Shot evidence not found: "
                f"{evidence_path}"
            )

        evidence = self._load_json(
            evidence_path
        )

        shots = evidence.get(
            "shots",
            [],
        )

        if not shots:
            raise CandidateDetectionError(
                "No shots found in shot_evidence.json"
            )

        print()
        print("=" * 70)
        print("COMMERCIAL CANDIDATE DETECTION")
        print("=" * 70)
        print(
            f"Video               : {video_path}"
        )
        print(
            f"Shots               : {len(shots)}"
        )
        print(
            f"Candidate threshold : "
            f"{self.candidate_threshold:.2f}"
        )
        print(
            f"Bridge threshold    : "
            f"{self.bridge_threshold:.2f}"
        )
        print(
            f"Maximum bridge gap  : "
            f"{self.max_bridge_gap:.2f}s"
        )
        print("=" * 70)

        scored_shots = []

        # -----------------------------------------------------
        # Score each shot
        # -----------------------------------------------------

        for shot in shots:

            scored = self._score_shot(
                shot
            )

            scored_shots.append(
                scored
            )

        # -----------------------------------------------------
        # Add neighbor/context effects
        # -----------------------------------------------------

        self._apply_neighbor_context(
            scored_shots
        )

        # -----------------------------------------------------
        # Select candidates
        # -----------------------------------------------------

        candidates = [
            shot
            for shot in scored_shots
            if shot["candidate_score"]
            >= self.candidate_threshold
        ]

        # -----------------------------------------------------
        # Bridge candidate regions
        # -----------------------------------------------------

        regions = self._build_regions(
            candidates
        )

        # -----------------------------------------------------
        # Print results
        # -----------------------------------------------------

        self._print_results(
            scored_shots,
            candidates,
            regions,
        )

        output = {
            "schema_version": "1.0",

            "source": {
                "video": str(
                    video_path
                ),
                "evidence": str(
                    evidence_path
                ),
            },

            "strategy": {
                "type": "lightweight_evidence_scoring",
                "candidate_threshold": (
                    self.candidate_threshold
                ),
                "bridge_threshold": (
                    self.bridge_threshold
                ),
                "max_bridge_gap": (
                    self.max_bridge_gap
                ),
            },

            "shots": scored_shots,

            "candidate_shots": [
                int(
                    shot["shot_index"]
                )
                for shot in candidates
            ],

            "candidate_regions": regions,

            "stats": {
                "total_shots": len(
                    scored_shots
                ),
                "candidate_shots": len(
                    candidates
                ),
                "candidate_regions": len(
                    regions
                ),
            },
        }

        output_path = (
            output_dir
            / "candidates.json"
        )

        self._save_json(
            output,
            output_path,
        )

        print()
        print(
            f"Candidate results saved: "
            f"{output_path}"
        )

        return output

    # =========================================================
    # SCORE SHOT
    # =========================================================

    def _score_shot(
        self,
        shot: dict,
    ) -> dict:

        shot_index = int(
            shot["shot_index"]
        )

        start = float(
            shot["start"]
        )

        end = float(
            shot["end"]
        )

        duration = max(
            0.0,
            end - start,
        )

        visual = shot.get(
            "visual",
            {},
        )

        audio = shot.get(
            "audio",
            {},
        )

        # -----------------------------------------------------
        # Extract evidence
        # -----------------------------------------------------

        ocr_text = self._extract_text(
            shot,
            visual,
            "ocr_text",
        )

        transcript = self._extract_text(
            shot,
            audio,
            "transcript",
        )

        combined = (
            f"{ocr_text} "
            f"{transcript}"
        ).lower()

        signals = []
        score = 0.0

        # -----------------------------------------------------
        # Strong CTA
        # -----------------------------------------------------

        strong_matches = self._find_keywords(
            combined,
            self.STRONG_KEYWORDS,
        )

        if strong_matches:

            score += min(
                0.40,
                0.20
                * len(strong_matches),
            )

            for keyword in strong_matches:

                signals.append(
                    {
                        "type": "strong_cta",
                        "value": keyword,
                        "weight": 0.20,
                    }
                )

        # -----------------------------------------------------
        # Commercial keywords
        # -----------------------------------------------------

        commercial_matches = self._find_keywords(
            combined,
            self.COMMERCIAL_KEYWORDS,
        )

        if commercial_matches:

            score += min(
                0.30,
                0.10
                * len(commercial_matches),
            )

            for keyword in commercial_matches:

                signals.append(
                    {
                        "type": "commercial_keyword",
                        "value": keyword,
                        "weight": 0.10,
                    }
                )

        # -----------------------------------------------------
        # Promotional phrases
        # -----------------------------------------------------

        promotional_matches = self._find_keywords(
            combined,
            self.PROMOTIONAL_PHRASES,
        )

        if promotional_matches:

            score += min(
                0.30,
                0.15
                * len(promotional_matches),
            )

            for phrase in promotional_matches:

                signals.append(
                    {
                        "type": "promotional_phrase",
                        "value": phrase,
                        "weight": 0.15,
                    }
                )

        # -----------------------------------------------------
        # Phone number
        # -----------------------------------------------------

        if self._contains_phone_number(
            combined
        ):

            score += 0.30

            signals.append(
                {
                    "type": "contact_information",
                    "value": "phone number",
                    "weight": 0.30,
                }
            )

        # -----------------------------------------------------
        # Price
        # -----------------------------------------------------

        if self._contains_price(
            combined
        ):

            score += 0.25

            signals.append(
                {
                    "type": "commercial_price",
                    "value": "price/currency",
                    "weight": 0.25,
                }
            )

        # -----------------------------------------------------
        # Business information
        # -----------------------------------------------------

        business_matches = self._find_keywords(
            combined,
            self.BUSINESS_TERMS,
        )

        if business_matches:

            score += min(
                0.20,
                0.05
                * len(business_matches),
            )

            for term in business_matches:

                signals.append(
                    {
                        "type": "business_information",
                        "value": term,
                        "weight": 0.05,
                    }
                )

        # -----------------------------------------------------
        # Commercial visual evidence from existing OCR
        # -----------------------------------------------------

        if ocr_text:

            # OCR is especially valuable because it comes
            # directly from the visual layer.
            score += 0.05

            signals.append(
                {
                    "type": "ocr_present",
                    "value": "OCR text detected",
                    "weight": 0.05,
                }
            )

        # -----------------------------------------------------
        # Cap score
        # -----------------------------------------------------

        score = min(
            1.0,
            score,
        )

        return {
            "shot_index": shot_index,
            "start": start,
            "end": end,
            "duration": round(
                duration,
                3,
            ),

            "ocr_text": ocr_text,

            "transcript": transcript,

            "candidate_score": round(
                score,
                3,
            ),

            "base_score": round(
                score,
                3,
            ),

            "context_bonus": 0.0,

            "signals": signals,

            "candidate": (
                score
                >= self.candidate_threshold
            ),
        }

    # =========================================================
    # NEIGHBOR CONTEXT
    # =========================================================

    def _apply_neighbor_context(
        self,
        shots: list,
    ) -> None:
        """
        Commercial content often spans several adjacent shots.

        Give a modest bonus to shots immediately adjacent to
        strong candidates.

        This is deliberately small so neighboring shots do not
        automatically become advertisements.
        """

        for index, shot in enumerate(
            shots
        ):

            neighbors = []

            if index > 0:

                neighbors.append(
                    shots[index - 1]
                )

            if index + 1 < len(shots):

                neighbors.append(
                    shots[index + 1]
                )

            neighbor_scores = [
                neighbor[
                    "base_score"
                ]
                for neighbor in neighbors
            ]

            if not neighbor_scores:
                continue

            strongest_neighbor = max(
                neighbor_scores
            )

            if (
                strongest_neighbor
                >= self.candidate_threshold
            ):

                # Only bridge reasonably close shots.
                close_neighbor = False

                for neighbor in neighbors:

                    gap = self._gap_between(
                        shot,
                        neighbor,
                    )

                    if gap <= self.max_bridge_gap:

                        close_neighbor = True
                        break

                if close_neighbor:

                    bonus = min(
                        0.15,
                        strongest_neighbor
                        * 0.15,
                    )

                    shot[
                        "context_bonus"
                    ] = round(
                        bonus,
                        3,
                    )

                    shot[
                        "candidate_score"
                    ] = round(
                        min(
                            1.0,
                            shot[
                                "base_score"
                            ]
                            + bonus,
                        ),
                        3,
                    )

                    shot[
                        "candidate"
                    ] = (
                        shot[
                            "candidate_score"
                        ]
                        >= self.candidate_threshold
                    )

                    shot[
                        "signals"
                    ].append(
                        {
                            "type": "neighbor_context",
                            "value": (
                                "adjacent commercial "
                                "candidate"
                            ),
                            "weight": round(
                                bonus,
                                3,
                            ),
                        }
                    )

    # =========================================================
    # BUILD REGIONS
    # =========================================================

    def _build_regions(
        self,
        candidates: list,
    ) -> list:
        """
        Merge nearby candidate shots into larger regions.

        This prevents tiny 0.2-second shots from becoming
        independent VLM jobs.
        """

        if not candidates:
            return []

        ordered = sorted(
            candidates,
            key=lambda shot: shot[
                "start"
            ],
        )

        regions = []

        current = {
            "start": ordered[0][
                "start"
            ],
            "end": ordered[0][
                "end"
            ],
            "shots": [
                int(
                    ordered[0][
                        "shot_index"
                    ]
                )
            ],
            "max_score": ordered[0][
                "candidate_score"
            ],
        }

        for shot in ordered[1:]:

            gap = (
                shot["start"]
                - current["end"]
            )

            if gap <= self.max_bridge_gap:

                current["end"] = max(
                    current["end"],
                    shot["end"],
                )

                current["shots"].append(
                    int(
                        shot["shot_index"]
                    )
                )

                current["max_score"] = max(
                    current["max_score"],
                    shot["candidate_score"],
                )

            else:

                regions.append(
                    self._finalize_region(
                        current
                    )
                )

                current = {
                    "start": shot[
                        "start"
                    ],
                    "end": shot[
                        "end"
                    ],
                    "shots": [
                        int(
                            shot[
                                "shot_index"
                            ]
                        )
                    ],
                    "max_score": shot[
                        "candidate_score"
                    ],
                }

        regions.append(
            self._finalize_region(
                current
            )
        )

        return regions

    # =========================================================
    # FINALIZE REGION
    # =========================================================

    @staticmethod
    def _finalize_region(
        region: dict,
    ) -> dict:

        start = float(
            region["start"]
        )

        end = float(
            region["end"]
        )

        return {
            "start": round(
                start,
                3,
            ),
            "end": round(
                end,
                3,
            ),
            "duration": round(
                end - start,
                3,
            ),
            "shots": region[
                "shots"
            ],
            "max_score": round(
                region["max_score"],
                3,
            ),
        }

    # =========================================================
    # TEXT EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_text(
        shot: dict,
        container: dict,
        key: str,
    ) -> str:

        value = container.get(
            key,
            "",
        )

        if value is None:

            return ""

        return str(
            value
        ).strip()

    # =========================================================
    # KEYWORD MATCHING
    # =========================================================

    @staticmethod
    def _find_keywords(
        text: str,
        keywords: set,
    ) -> list:

        matches = []

        for keyword in keywords:

            if keyword in text:

                matches.append(
                    keyword
                )

        return sorted(
            matches,
            key=len,
            reverse=True,
        )

    # =========================================================
    # PHONE DETECTION
    # =========================================================

    @staticmethod
    def _contains_phone_number(
        text: str,
    ) -> bool:

        # Flexible enough for:
        # +91-9100708888
        # 9100708888
        # 09100708888
        # 91007 08888

        matches = re.findall(
            r"(?:\+?\d[\d\s().-]{7,}\d)",
            text,
        )

        for match in matches:

            digits = re.sub(
                r"\D",
                "",
                match,
            )

            if 8 <= len(
                digits
            ) <= 15:

                return True

        return False

    # =========================================================
    # PRICE DETECTION
    # =========================================================

    @staticmethod
    def _contains_price(
        text: str,
    ) -> bool:

        currency_symbols = {
            "₹",
            "$",
            "€",
            "£",
            "¥",
        }

        if any(
            symbol in text
            for symbol in currency_symbols
        ):

            return True

        patterns = [
            r"\binr\b",
            r"\brs\.?\b",
            r"\busd\b",
            r"\beur\b",
            r"\bprice\b",
            r"\bprices\b",
            r"\bcost\b",
            r"\bcosts\b",
            r"\bstarting from\b",
        ]

        return any(
            re.search(
                pattern,
                text,
            )
            for pattern in patterns
        )

    # =========================================================
    # GAP
    # =========================================================

    @staticmethod
    def _gap_between(
        first: dict,
        second: dict,
    ) -> float:

        if first["end"] < second["start"]:

            return second[
                "start"
            ] - first["end"]

        if second["end"] < first["start"]:

            return first[
                "start"
            ] - second["end"]

        return 0.0

    # =========================================================
    # LOAD JSON
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

            raise CandidateDetectionError(
                f"Could not read: {path}"
            ) from error

        except json.JSONDecodeError as error:

            raise CandidateDetectionError(
                f"Invalid JSON: {path}"
            ) from error

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

    # =========================================================
    # PRINT RESULTS
    # =========================================================

    @staticmethod
    def _print_results(
        shots: list,
        candidates: list,
        regions: list,
    ) -> None:

        print()
        print("=" * 70)
        print("SHOT CANDIDATE SCORES")
        print("=" * 70)

        for shot in shots:

            state = (
                "CANDIDATE"
                if shot["candidate"]
                else "skip"
            )

            print(
                f"Shot {shot['shot_index']:02d} | "
                f"{shot['start']:.3f}s -> "
                f"{shot['end']:.3f}s | "
                f"score={shot['candidate_score']:.3f} | "
                f"{state}"
            )

            if shot["signals"]:

                signal_text = ", ".join(
                    signal["value"]
                    for signal in shot[
                        "signals"
                    ]
                )

                print(
                    f"  Signals: {signal_text}"
                )

        print()
        print("=" * 70)
        print("CANDIDATE REGIONS")
        print("=" * 70)

        if not regions:

            print(
                "No candidate regions found."
            )

        else:

            for index, region in enumerate(
                regions,
                start=1,
            ):

                print(
                    f"Region {index:02d}: "
                    f"{region['start']:.3f}s -> "
                    f"{region['end']:.3f}s "
                    f"({region['duration']:.3f}s)"
                )

                print(
                    f"  Shots      : "
                    f"{region['shots']}"
                )

                print(
                    f"  Max score  : "
                    f"{region['max_score']:.3f}"
                )

        print()
        print("=" * 70)
        print("CANDIDATE SUMMARY")
        print("=" * 70)

        print(
            f"Total shots       : "
            f"{len(shots)}"
        )

        print(
            f"Candidate shots   : "
            f"{len(candidates)}"
        )

        print(
            f"Candidate regions : "
            f"{len(regions)}"
        )

        print("=" * 70)


# =============================================================
# COMMAND LINE
# =============================================================

def main():

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python -m src.candidate_detector "
            "<video_path>"
        )

        sys.exit(1)

    video_path = Path(
        sys.argv[1]
    )

    detector = CandidateDetector()

    try:

        detector.detect(
            video_path
        )

    except CandidateDetectionError as error:

        print()
        print(
            "CANDIDATE DETECTION FAILED"
        )

        print(error)

        sys.exit(1)


if __name__ == "__main__":
    main()
