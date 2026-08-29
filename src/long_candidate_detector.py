import json
import re
import sys
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATE_THRESHOLD = 0.20
MAX_BRIDGE_GAP = 30.0

# Audio gets its own contribution.
TRANSCRIPT_PRESENT_SCORE = 0.05

# Commercial / advertising language.
COMMERCIAL_KEYWORDS = {
    "buy": 0.25,
    "shop": 0.25,
    "order": 0.25,
    "book now": 0.35,
    "contact us": 0.35,
    "call now": 0.35,
    "visit us": 0.30,
    "learn more": 0.25,
    "sign up": 0.25,
    "subscribe": 0.20,
    "download": 0.20,
    "offer": 0.20,
    "limited time": 0.30,
    "discount": 0.30,
    "sale": 0.25,
    "free": 0.15,
    "coupon": 0.30,
    "promo": 0.30,
    "promotion": 0.30,
    "deal": 0.20,
    "price": 0.15,
    "only": 0.10,
    "available now": 0.25,
    "website": 0.15,
    "www": 0.20,
    ".com": 0.20,
    "instagram": 0.10,
    "facebook": 0.10,
    "youtube": 0.10,
    "whatsapp": 0.15,
}

# Common advertising phrases.
AD_PHRASES = [
    "for more information",
    "call us",
    "contact us",
    "visit us",
    "book your",
    "book now",
    "order now",
    "buy now",
    "get yours",
    "don't miss",
    "special offer",
    "special price",
    "limited offer",
    "limited time",
    "available now",
    "available today",
    "click here",
    "follow us",
    "message us",
    "dm us",
]

# Strong commercial patterns.
PHONE_PATTERN = re.compile(
    r"(?:\+?\d[\d\s().-]{7,}\d)"
)

URL_PATTERN = re.compile(
    r"(?:https?://|www\.|"
    r"[a-zA-Z0-9-]+\.(?:com|in|co|net|org))",
    re.IGNORECASE,
)

MONEY_PATTERN = re.compile(
    r"(?:₹|\$|€|£)\s?\d+"
)

PERCENT_PATTERN = re.compile(
    r"\b\d{1,3}\s?%\b"
)


class LongCandidateDetectorError(Exception):
    """Raised when long-form candidate detection fails."""


# ============================================================
# JSON LOADING
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        raise LongCandidateDetectorError(
            f"Missing JSON file: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as error:
        raise LongCandidateDetectorError(
            f"Invalid JSON file: {path}: {error}"
        ) from error


# ============================================================
# OCR
# ============================================================

def extract_ocr(image_path: str) -> str:
    """
    Run lightweight Tesseract OCR.

    OCR failure does not stop the pipeline.
    """

    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    try:
        image = Image.open(
            image_path
        ).convert("RGB")

        text = pytesseract.image_to_string(
            image,
            config="--psm 6",
        )

        return " ".join(
            text.replace("\n", " ").split()
        )

    except Exception:
        return ""


# ============================================================
# ASR LOADING
# ============================================================

def load_transcript(
    transcript_path: Path,
) -> list[dict[str, Any]]:
    """
    Load Whisper transcript segments.

    Supports:
        {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "..."
                }
            ]
        }

    Also supports a raw list of segments.
    """

    if not transcript_path.exists():
        return []

    try:
        data = load_json(
            transcript_path
        )

    except LongCandidateDetectorError:
        return []

    if isinstance(data, list):
        raw_segments = data

    elif isinstance(data, dict):
        raw_segments = data.get(
            "segments",
            [],
        )

    else:
        return []

    result = []

    for segment in raw_segments:

        if not isinstance(
            segment,
            dict,
        ):
            continue

        try:
            start = float(
                segment.get(
                    "start",
                    0.0,
                )
            )

            end = float(
                segment.get(
                    "end",
                    start,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        text = str(
            segment.get(
                "text",
                "",
            )
            or ""
        ).strip()

        if not text:
            continue

        result.append(
            {
                "start": start,
                "end": end,
                "text": text,
            }
        )

    result.sort(
        key=lambda x: x["start"]
    )

    return result


# ============================================================
# TRANSCRIPT ALIGNMENT
# ============================================================

def transcript_for_region(
    start: float,
    end: float,
    transcript_segments: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Return all ASR segments that overlap the region.

    Overlap rule:

        segment_end > region_start
        AND
        segment_start < region_end
    """

    matched = []

    for segment in transcript_segments:

        segment_start = segment["start"]
        segment_end = segment["end"]

        if (
            segment_end > start
            and segment_start < end
        ):
            matched.append(
                segment
            )

    text = " ".join(
        segment["text"]
        for segment in matched
    ).strip()

    return text, matched


# ============================================================
# TEXT SCORING
# ============================================================

def score_text(
    text: str,
) -> tuple[float, list[str]]:
    """
    Score OCR/transcript text for commercial evidence.
    """

    if not text:
        return 0.0, []

    normalized = (
        text.lower()
        .replace("\n", " ")
    )

    score = 0.0
    signals: list[str] = []

    # --------------------------------------------------------
    # Commercial keywords
    # --------------------------------------------------------

    for keyword, weight in (
        COMMERCIAL_KEYWORDS.items()
    ):

        if keyword in normalized:

            score += weight

            signals.append(
                f"commercial keyword: {keyword}"
            )

    # --------------------------------------------------------
    # Advertising phrases
    # --------------------------------------------------------

    for phrase in AD_PHRASES:

        if phrase in normalized:

            score += 0.25

            signals.append(
                f"advertising phrase: {phrase}"
            )

    # --------------------------------------------------------
    # Phone number
    # --------------------------------------------------------

    if PHONE_PATTERN.search(text):

        score += 0.25

        signals.append(
            "phone number"
        )

    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    if URL_PATTERN.search(text):

        score += 0.20

        signals.append(
            "website/url"
        )

    # --------------------------------------------------------
    # Currency
    # --------------------------------------------------------

    if MONEY_PATTERN.search(text):

        score += 0.15

        signals.append(
            "price/currency"
        )

    # --------------------------------------------------------
    # Percentage / discount
    # --------------------------------------------------------

    if PERCENT_PATTERN.search(text):

        score += 0.20

        signals.append(
            "percentage/discount"
        )

    # --------------------------------------------------------
    # Generic CTA combinations
    # --------------------------------------------------------

    cta_words = [
        "buy",
        "book",
        "order",
        "contact",
        "call",
        "visit",
        "shop",
        "download",
        "sign up",
        "follow",
        "message",
    ]

    cta_found = [
        word
        for word in cta_words
        if word in normalized
    ]

    if len(cta_found) >= 2:

        score += 0.20

        signals.append(
            "multiple CTA terms"
        )

    score = min(
        score,
        1.0,
    )

    signals = list(
        dict.fromkeys(
            signals
        )
    )

    return score, signals


# ============================================================
# REGION SCORING
# ============================================================

def score_region(
    region: dict[str, Any],
    transcript_segments: list[dict[str, Any]],
) -> dict[str, Any]:

    start = float(
        region["start"]
    )

    end = float(
        region["end"]
    )

    duration = float(
        region.get(
            "duration",
            end - start,
        )
    )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    ocr_text = str(
        region.get(
            "ocr_text",
            "",
        )
        or ""
    ).strip()

    if not ocr_text:

        frame = (
            region.get("frame")
            or {}
        )

        frame_path = frame.get(
            "path"
        )

        if frame_path:

            ocr_text = extract_ocr(
                frame_path
            )

    # --------------------------------------------------------
    # AUDIO / ASR
    # --------------------------------------------------------

    transcript, matched_asr = (
        transcript_for_region(
            start,
            end,
            transcript_segments,
        )
    )

    # --------------------------------------------------------
    # Score OCR independently
    # --------------------------------------------------------

    ocr_score, ocr_signals = (
        score_text(
            ocr_text
        )
    )

    # --------------------------------------------------------
    # Score transcript independently
    # --------------------------------------------------------

    transcript_score, transcript_signals = (
        score_text(
            transcript
        )
    )

    # --------------------------------------------------------
    # Combine evidence
    #
    # Do NOT simply add both full scores.
    #
    # Instead:
    #   visual evidence + audio evidence
    #
    # This prevents one noisy OCR result and one
    # noisy transcript from immediately producing
    # a huge score.
    # --------------------------------------------------------

    score = min(
        1.0,
        max(
            ocr_score,
            transcript_score,
        )
        + (
            min(
                ocr_score,
                transcript_score,
            )
            * 0.50
        ),
    )

    # Audio alone can provide weak evidence.
    if transcript and transcript_score == 0:

        score = max(
            score,
            TRANSCRIPT_PRESENT_SCORE,
        )

    signals = []

    for signal in ocr_signals:

        signals.append(
            f"visual: {signal}"
        )

    for signal in transcript_signals:

        signals.append(
            f"audio: {signal}"
        )

    # --------------------------------------------------------
    # Evidence presence
    # --------------------------------------------------------

    if ocr_text:

        signals.append(
            "OCR text detected"
        )

    if transcript:

        signals.append(
            "speech/transcript detected"
        )

    if matched_asr:

        signals.append(
            f"ASR segments: {len(matched_asr)}"
        )

    signals = list(
        dict.fromkeys(
            signals
        )
    )

    return {
        "region_index": region[
            "region_index"
        ],
        "start": start,
        "end": end,
        "duration": duration,
        "score": round(
            score,
            3,
        ),
        "candidate": (
            score >= CANDIDATE_THRESHOLD
        ),
        "signals": signals,
        "ocr_text": ocr_text,
        "transcript": transcript,
        "asr_segments": matched_asr,
        "frame": region.get(
            "frame"
        ),
    }


# ============================================================
# MERGE CANDIDATE REGIONS
# ============================================================

def build_regions(
    scored_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    candidates = [
        region
        for region in scored_regions
        if region["candidate"]
    ]

    if not candidates:
        return []

    candidates.sort(
        key=lambda x: x["start"]
    )

    merged = []

    current = {
        "start": candidates[0]["start"],
        "end": candidates[0]["end"],
        "regions": [
            candidates[0]["region_index"]
        ],
        "max_score": candidates[0]["score"],
    }

    for region in candidates[1:]:

        gap = (
            region["start"]
            - current["end"]
        )

        if gap <= MAX_BRIDGE_GAP:

            current["end"] = max(
                current["end"],
                region["end"],
            )

            current[
                "regions"
            ].append(
                region["region_index"]
            )

            current[
                "max_score"
            ] = max(
                current["max_score"],
                region["score"],
            )

        else:

            merged.append(
                current
            )

            current = {
                "start": region["start"],
                "end": region["end"],
                "regions": [
                    region[
                        "region_index"
                    ]
                ],
                "max_score": region[
                    "score"
                ],
            }

    merged.append(
        current
    )

    result = []

    for index, region in enumerate(
        merged,
        start=1,
    ):

        result.append(
            {
                "candidate_index": index,
                "start": round(
                    region["start"],
                    3,
                ),
                "end": round(
                    region["end"],
                    3,
                ),
                "duration": round(
                    region["end"]
                    - region["start"],
                    3,
                ),
                "source_regions": region[
                    "regions"
                ],
                "max_score": round(
                    region["max_score"],
                    3,
                ),
            }
        )

    return result


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect(
    video_path: str | Path,
) -> dict[str, Any]:

    video_path = Path(
        video_path
    )

    output_dir = (
        Path("data")
        / "outputs"
        / video_path.stem
    )

    evidence_path = (
        output_dir
        / "long_evidence.json"
    )

    transcript_path = (
        output_dir
        / "transcript.json"
    )

    # --------------------------------------------------------
    # Load evidence
    # --------------------------------------------------------

    evidence = load_json(
        evidence_path
    )

    regions = evidence.get(
        "regions",
        [],
    )

    if not regions:

        raise LongCandidateDetectorError(
            "No regions found in "
            "long_evidence.json"
        )

    # --------------------------------------------------------
    # Load ASR
    # --------------------------------------------------------

    transcript_segments = (
        load_transcript(
            transcript_path
        )
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "LONG-FORM COMMERCIAL "
        "CANDIDATE DETECTION"
    )
    print("=" * 70)

    print(
        f"Video               : "
        f"{video_path}"
    )

    print(
        f"Regions             : "
        f"{len(regions)}"
    )

    print(
        f"ASR segments        : "
        f"{len(transcript_segments)}"
    )

    print(
        f"Candidate threshold : "
        f"{CANDIDATE_THRESHOLD:.2f}"
    )

    print(
        f"Bridge gap          : "
        f"{MAX_BRIDGE_GAP:.1f}s"
    )

    print("=" * 70)

    if transcript_segments:

        print(
            "Audio evidence      : ENABLED"
        )

    else:

        print(
            "Audio evidence      : "
            "NOT AVAILABLE"
        )

    # --------------------------------------------------------
    # Score regions
    # --------------------------------------------------------

    scored_regions = []

    print()
    print("=" * 70)
    print(
        "REGION CANDIDATE SCORES"
    )
    print("=" * 70)

    for position, region in enumerate(
        regions,
        start=1,
    ):

        scored = score_region(
            region,
            transcript_segments,
        )

        scored_regions.append(
            scored
        )

        status = (
            "CANDIDATE"
            if scored["candidate"]
            else "skip"
        )

        print(
            f"[{position}/{len(regions)}] "
            f"Region "
            f"{scored['region_index']:03d} | "
            f"{scored['start']:.1f}s -> "
            f"{scored['end']:.1f}s | "
            f"score="
            f"{scored['score']:.3f} | "
            f"{status}"
        )

        if scored["signals"]:

            print(
                "  Signals: "
                + ", ".join(
                    scored["signals"]
                )
            )

        if scored["transcript"]:

            preview = (
                scored["transcript"][:150]
                .replace(
                    "\n",
                    " ",
                )
            )

            print(
                f"  ASR: {preview}"
            )

        if scored["ocr_text"]:

            preview = (
                scored["ocr_text"][:150]
                .replace(
                    "\n",
                    " ",
                )
            )

            print(
                f"  OCR: {preview}"
            )

    # --------------------------------------------------------
    # Candidate regions
    # --------------------------------------------------------

    raw_candidates = [
        region
        for region in scored_regions
        if region["candidate"]
    ]

    merged_regions = build_regions(
        scored_regions
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {

        "schema_version": "1.0",

        "source": {
            "video": str(
                video_path
            ),
            "long_evidence": str(
                evidence_path
            ),
            "transcript": (
                str(transcript_path)
                if transcript_path.exists()
                else None
            ),
        },

        "method": {

            "candidate_threshold":
                CANDIDATE_THRESHOLD,

            "max_bridge_gap":
                MAX_BRIDGE_GAP,

            "ocr":
                "tesseract",

            "asr":
                "transcript.json",

            "alignment":
                "timestamp_overlap",

            "multimodal":
                True,
        },

        "regions":
            scored_regions,

        "candidates":
            merged_regions,

        # Keep this alias for compatibility
        # with earlier pipeline components.
        "candidate_regions":
            merged_regions,

        "stats": {

            "total_regions":
                len(regions),

            "asr_segments":
                len(
                    transcript_segments
                ),

            "candidate_regions_raw":
                len(
                    raw_candidates
                ),

            "candidate_regions_merged":
                len(
                    merged_regions
                ),

            "audio_enabled":
                bool(
                    transcript_segments
                ),
        },
    }

    output_path = (
        output_dir
        / "long_candidates.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "CANDIDATE REGIONS"
    )
    print("=" * 70)

    if not merged_regions:

        print(
            "No commercial candidate "
            "regions found."
        )

    else:

        for candidate in merged_regions:

            print(
                f"Candidate "
                f"{candidate['candidate_index']:02d}: "
                f"{candidate['start']:.3f}s -> "
                f"{candidate['end']:.3f}s "
                f"("
                f"{candidate['duration']:.3f}s"
                f")"
            )

            print(
                f"  Source regions : "
                f"{candidate['source_regions']}"
            )

            print(
                f"  Max score      : "
                f"{candidate['max_score']:.3f}"
            )

    print()
    print("=" * 70)
    print(
        "CANDIDATE SUMMARY"
    )
    print("=" * 70)

    print(
        f"Total regions          : "
        f"{len(regions)}"
    )

    print(
        f"ASR segments           : "
        f"{len(transcript_segments)}"
    )

    print(
        f"Raw candidate regions  : "
        f"{len(raw_candidates)}"
    )

    print(
        f"Merged candidate areas : "
        f"{len(merged_regions)}"
    )

    print(
        f"Audio evidence         : "
        f"{'YES' if transcript_segments else 'NO'}"
    )

    print()
    print(
        f"Results saved: "
        f"{output_path}"
    )

    return output


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "  python -m "
            "src.long_candidate_detector "
            "<video_path>"
        )

        sys.exit(1)

    video_path = sys.argv[1]

    try:

        detect(
            video_path
        )

    except LongCandidateDetectorError as error:

        print()
        print(
            "LONG CANDIDATE DETECTION FAILED"
        )

        print(error)

        sys.exit(1)

    except KeyboardInterrupt:

        print()
        print(
            "Detection interrupted."
        )

        sys.exit(130)