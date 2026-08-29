import base64
import json
import re
import sys
import time
from pathlib import Path

import requests


# =============================================================
# UTF-8 SUPPORT
# =============================================================

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


# =============================================================
# ERROR
# =============================================================

class VLMClassificationError(Exception):
    """Raised when VLM classification fails."""


# =============================================================
# VLM CLASSIFIER
# =============================================================

class VLMClassifier:
    """
    CPU-optimized Qwen2.5-VL classifier.

    Strategy:

        shot_evidence.json
                |
                v
        commercial evidence gate
                |
        +-------+-------+
        |               |
      SKIP             VLM
        |               |
        v               v
     organic       Qwen2.5-VL
                        |
                        v
              advertisement /
              organic_content /
              uncertain

    CPU optimization:
        - Only middle frame is sent
        - Only commercially suspicious shots reach VLM
        - Short prompt
        - Small output budget
        - Ollama keep_alive
        - No regional / multi-frame VLM calls
    """

    # =========================================================
    # MODEL CONFIG
    # =========================================================

    DEFAULT_MODEL = "qwen2.5vl:7b"

    DEFAULT_OLLAMA_URL = (
        "http://localhost:11434/api/chat"
    )

    # Your CPU can take a long time, but requests should not
    # hang forever.
    DEFAULT_TIMEOUT = 900

    # Keep model loaded between calls.
    DEFAULT_KEEP_ALIVE = "10m"

    # Small because we only need structured JSON.
    DEFAULT_NUM_PREDICT = 180

    # =========================================================
    # COMMERCIAL KEYWORDS
    # =========================================================

    COMMERCIAL_KEYWORDS = {
        # CTA
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
        "subscribe",
        "follow us",
        "follow for more",

        # Commercial
        "offer",
        "discount",
        "sale",
        "deal",
        "price",
        "cost",
        "starting from",
        "limited time",
        "free",
        "special offer",

        # Purchase/service
        "appointment",
        "booking",
        "delivery",
        "purchase",
        "order",
        "clinic",
        "hospital",
        "dentist",
        "dental",

        # Promotional
        "for more",
        "our services",
        "our products",
        "services",
        "product",
        "promo",
        "promotion",
    }

    STRONG_COMMERCIAL_KEYWORDS = {
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
        "discount",
        "sale",
        "offer",
        "starting from",
        "limited time",
        "special offer",
    }

    BUSINESS_TERMS = {
        "clinic",
        "hospital",
        "doctor",
        "dr.",
        "dentist",
        "dental",
        "address",
        "road no",
        "hyderabad",
        "jubilee hills",
    }

    # =========================================================
    # INIT
    # =========================================================

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
            raise VLMClassificationError(
                f"Video not found: {video_path}"
            )

        video_name = video_path.stem

        output_dir = (
            Path("data")
            / "outputs"
            / video_name
        )

        frames_dir = (
            Path("data")
            / "frames"
            / "shots"
            / video_name
            / video_name
        )

        evidence_path = (
            output_dir
            / "shot_evidence.json"
        )

        if not evidence_path.exists():
            raise VLMClassificationError(
                f"Shot evidence not found: {evidence_path}"
            )

        evidence = self._load_json(
            evidence_path
        )

        shots = evidence.get(
            "shots",
            [],
        )

        if not shots:
            raise VLMClassificationError(
                "No shots found in shot_evidence.json"
            )

        # -----------------------------------------------------
        # Header
        # -----------------------------------------------------

        print()
        print("=" * 70)
        print("CPU-OPTIMIZED VLM CLASSIFICATION")
        print("=" * 70)
        print(
            f"Model       : {self.model}"
        )
        print(
            f"Video       : {video_path}"
        )
        print(
            f"Shots       : {len(shots)}"
        )
        print(
            "Frame mode  : middle representative"
        )
        print(
            "VLM mode    : evidence gated"
        )
        print(
            f"Output max  : {self.num_predict} tokens"
        )
        print("=" * 70)

        # -----------------------------------------------------
        # Evidence gate
        # -----------------------------------------------------

        candidates = []

        for shot in shots:

            gate = self._commercial_evidence_gate(
                shot
            )

            shot["_vlm_gate"] = gate

            if gate["needs_vlm"]:
                candidates.append(shot)

        print()
        print("=" * 70)
        print("VLM EVIDENCE GATE")
        print("=" * 70)

        print(
            f"Total shots       : {len(shots)}"
        )

        print(
            f"VLM candidates    : {len(candidates)}"
        )

        print(
            f"VLM skipped       : "
            f"{len(shots) - len(candidates)}"
        )

        print("=" * 70)

        # -----------------------------------------------------
        # Process shots
        # -----------------------------------------------------

        results = []

        vlm_calls = 0

        total_vlm_time = 0.0

        for position, shot in enumerate(
            shots,
            start=1,
        ):

            shot_index = int(
                shot["shot_index"]
            )

            gate = shot["_vlm_gate"]

            print()
            print(
                f"[{position}/{len(shots)}] "
                f"Shot {shot_index:02d} "
                f"| "
                f"{shot['start']:.3f}s -> "
                f"{shot['end']:.3f}s"
            )

            if gate["signals"]:
                print(
                    "  Signals: "
                    + ", ".join(
                        gate["signals"]
                    )
                )

            # -------------------------------------------------
            # SKIP
            # -------------------------------------------------

            if not gate["needs_vlm"]:

                result = (
                    self._build_skipped_result(
                        shot,
                        gate,
                    )
                )

                results.append(result)

                print(
                    "  Gate           : SKIP"
                )

                print(
                    "  Classification : "
                    "organic_content"
                )

                continue

            # -------------------------------------------------
            # VLM
            # -------------------------------------------------

            print(
                "  Gate           : VLM"
            )

            print(
                "  Sending middle frame..."
            )

            start_time = time.perf_counter()

            try:

                result = self.classify_shot(
                    shot=shot,
                    frames_dir=frames_dir,
                )

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                total_vlm_time += elapsed
                vlm_calls += 1

                result["gate"] = gate

                results.append(result)

                print(
                    f"  Classification : "
                    f"{result['classification']}"
                )

                print(
                    f"  Confidence     : "
                    f"{result['confidence']:.3f}"
                )

                print(
                    f"  Ad type        : "
                    f"{result.get('ad_type', 'other')}"
                )

                print(
                    f"  Brand          : "
                    f"{result.get('brand') or 'unknown'}"
                )

                print(
                    f"  VLM time       : "
                    f"{elapsed:.1f}s"
                )

            except Exception as error:

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                total_vlm_time += elapsed
                vlm_calls += 1

                print(
                    f"  VLM ERROR      : "
                    f"{error}"
                )

                results.append(
                    self._build_error_result(
                        shot,
                        gate,
                        error,
                    )
                )

        # -----------------------------------------------------
        # Remove internal gate
        # -----------------------------------------------------

        for shot in shots:
            shot.pop(
                "_vlm_gate",
                None,
            )

        # -----------------------------------------------------
        # Build output
        # -----------------------------------------------------

        output = self._build_output(
            video_path=video_path,
            results=results,
            total_shots=len(shots),
            vlm_calls=vlm_calls,
            total_vlm_time=total_vlm_time,
        )

        output_path = (
            output_dir
            / "vlm_classifications.json"
        )

        self._save_json(
            output,
            output_path,
        )

        self._print_summary(
            results=results,
            vlm_calls=vlm_calls,
            total_shots=len(shots),
            total_vlm_time=total_vlm_time,
        )

        print()
        print(
            f"Results saved: {output_path}"
        )

        return output

    # =========================================================
    # EVIDENCE GATE
    # =========================================================

    def _commercial_evidence_gate(
        self,
        shot: dict,
    ) -> dict:

        visual = shot.get(
            "visual",
            {},
        )

        audio = shot.get(
            "audio",
            {},
        )

        ocr_text = str(
            visual.get(
                "ocr_text",
                "",
            )
        ).strip()

        transcript = str(
            audio.get(
                "transcript",
                "",
            )
        ).strip()

        combined = (
            f"{ocr_text} "
            f"{transcript}"
        ).lower()

        signals = []

        # -----------------------------------------------------
        # Strong keywords
        # -----------------------------------------------------

        for keyword in sorted(
            self.STRONG_COMMERCIAL_KEYWORDS
        ):

            if keyword in combined:

                signal = (
                    f"commercial keyword: "
                    f"{keyword}"
                )

                if signal not in signals:
                    signals.append(signal)

        # -----------------------------------------------------
        # General keywords
        # -----------------------------------------------------

        for keyword in sorted(
            self.COMMERCIAL_KEYWORDS
        ):

            if keyword in combined:

                signal = (
                    f"commercial keyword: "
                    f"{keyword}"
                )

                if signal not in signals:
                    signals.append(signal)

        # -----------------------------------------------------
        # Phone number
        # -----------------------------------------------------

        if self._contains_phone_number(
            combined
        ):

            signals.append(
                "phone number"
            )

        # -----------------------------------------------------
        # Price
        # -----------------------------------------------------

        if self._contains_price(
            combined
        ):

            signals.append(
                "price/currency"
            )

        # -----------------------------------------------------
        # Business information
        # -----------------------------------------------------

        business_matches = [
            term
            for term in self.BUSINESS_TERMS
            if term in combined
        ]

        if business_matches:

            signals.append(
                "business/contact information"
            )

        # -----------------------------------------------------
        # Decision
        # -----------------------------------------------------

        needs_vlm = len(signals) > 0

        return {
            "needs_vlm": needs_vlm,
            "signals": signals,
            "reason": (
                "Commercial/promotional "
                "evidence detected."
                if needs_vlm
                else
                "No meaningful commercial "
                "evidence found in OCR or ASR."
            ),
        }

    # =========================================================
    # PHONE NUMBER
    # =========================================================

    @staticmethod
    def _contains_phone_number(
        text: str,
    ) -> bool:

        # Handles:
        #
        # +91-9100708888
        # 9100708888
        # 98765 43210
        # (040) 12345678

        groups = re.findall(
            r"\d[\d\s().+\-]{5,}\d",
            text,
        )

        for group in groups:

            digits = re.sub(
                r"\D",
                "",
                group,
            )

            if 7 <= len(digits) <= 15:
                return True

        return False

    # =========================================================
    # PRICE
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

        price_words = {
            " rs ",
            " rs.",
            " inr ",
            " usd ",
            " price ",
            " cost ",
            " costs ",
            " starting from ",
        }

        padded = (
            f" {text} "
        )

        return any(
            word in padded
            for word in price_words
        )

    # =========================================================
    # SKIPPED RESULT
    # =========================================================

    @staticmethod
    def _build_skipped_result(
        shot: dict,
        gate: dict,
    ) -> dict:

        visual = shot.get(
            "visual",
            {},
        )

        audio = shot.get(
            "audio",
            {},
        )

        return {
            "classification": "organic_content",
            "confidence": 0.80,
            "ad_type": "other",
            "brand": None,
            "reason": (
                "No meaningful commercial "
                "evidence was found in OCR "
                "or ASR, so the expensive "
                "VLM analysis was skipped."
            ),
            "signals": [],
            "shot_index": int(
                shot["shot_index"]
            ),
            "start": shot["start"],
            "end": shot["end"],
            "image": "",
            "ocr_text": visual.get(
                "ocr_text",
                "",
            ),
            "transcript": audio.get(
                "transcript",
                "",
            ),
            "vlm_skipped": True,
            "gate": gate,
        }

    # =========================================================
    # ERROR RESULT
    # =========================================================

    @staticmethod
    def _build_error_result(
        shot: dict,
        gate: dict,
        error: Exception,
    ) -> dict:

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
            "shot_index": int(
                shot["shot_index"]
            ),
            "start": shot["start"],
            "end": shot["end"],
            "image": "",
            "ocr_text": shot.get(
                "visual",
                {},
            ).get(
                "ocr_text",
                "",
            ),
            "transcript": shot.get(
                "audio",
                {},
            ).get(
                "transcript",
                "",
            ),
            "vlm_skipped": False,
            "error": str(error),
            "gate": gate,
        }

    # =========================================================
    # SINGLE SHOT VLM
    # =========================================================

    def classify_shot(
        self,
        shot: dict,
        frames_dir: Path,
    ) -> dict:

        shot_index = int(
            shot["shot_index"]
        )

        # -----------------------------------------------------
        # ONLY MIDDLE FRAME
        # -----------------------------------------------------

        image_path = (
            frames_dir
            / f"shot_{shot_index:02d}_middle.jpg"
        )

        if not image_path.exists():

            raise VLMClassificationError(
                f"Middle frame not found: "
                f"{image_path}"
            )

        print(
            f"  Image          : "
            f"{image_path}"
        )

        image_base64 = (
            self._encode_image(
                image_path
            )
        )

        # -----------------------------------------------------
        # SHORT PROMPT
        # -----------------------------------------------------

        prompt = self._build_prompt(
            shot
        )

        # -----------------------------------------------------
        # OLLAMA PAYLOAD
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

        # -----------------------------------------------------
        # REQUEST
        # -----------------------------------------------------

        response = self.session.post(
            self.ollama_url,
            json=payload,
            timeout=self.timeout,
        )

        if not response.ok:

            raise VLMClassificationError(
                "Ollama returned "
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )

        try:

            data = response.json()

        except ValueError as error:

            raise VLMClassificationError(
                "Ollama returned invalid JSON."
            ) from error

        raw_response = (
            data
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not raw_response:

            raise VLMClassificationError(
                "VLM returned an empty response."
            )

        # -----------------------------------------------------
        # PARSE
        # -----------------------------------------------------

        parsed = self._parse_json(
            raw_response
        )

        validated = self._validate_result(
            parsed
        )

        # -----------------------------------------------------
        # ADD PIPELINE METADATA
        # -----------------------------------------------------

        validated["shot_index"] = (
            shot_index
        )

        validated["start"] = (
            shot["start"]
        )

        validated["end"] = (
            shot["end"]
        )

        validated["image"] = (
            str(image_path)
        )

        validated["ocr_text"] = (
            shot
            .get("visual", {})
            .get("ocr_text", "")
        )

        validated["transcript"] = (
            shot
            .get("audio", {})
            .get("transcript", "")
        )

        validated["vlm_skipped"] = False

        return validated

    # =========================================================
    # SHORT CPU PROMPT
    # =========================================================

    @staticmethod
    def _build_prompt(
        shot: dict,
    ) -> str:

        visual = shot.get(
            "visual",
            {},
        )

        audio = shot.get(
            "audio",
            {},
        )

        ocr_text = str(
            visual.get(
                "ocr_text",
                "",
            )
        ).strip()

        transcript = str(
            audio.get(
                "transcript",
                "",
            )
        ).strip()

        if not ocr_text:
            ocr_text = "none"

        if not transcript:
            transcript = "none"

        # Keep OCR / ASR bounded.
        #
        # This prevents unexpectedly huge OCR strings
        # from consuming context.
        ocr_text = ocr_text[:1200]
        transcript = transcript[:800]

        return f"""
Classify this video shot.

Use the image plus OCR and transcript.

Advertisement means promotional/commercial intent such as:
- selling/promoting a product or service
- business promotion
- booking/order/purchase CTA
- contact information used for promotion
- price/discount/offer
- promotional social CTA

Do NOT call something an ad only because a person,
doctor, business, product, or logo appears.

Return ONLY JSON.

Schema:
{{
  "classification": "advertisement|organic_content|uncertain",
  "confidence": 0.0,
  "ad_type": "product_promotion|service_promotion|self_promo|affiliate|sponsor_read|product_placement|other",
  "brand": "brand name or null",
  "reason": "short evidence-based explanation",
  "signals": ["observable signal"]
}}

OCR:
{ocr_text}

ASR:
{transcript}
""".strip()

    # =========================================================
    # IMAGE ENCODING
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

            raise VLMClassificationError(
                f"Could not read image: "
                f"{path}"
            ) from error

    # =========================================================
    # JSON PARSING
    # =========================================================

    @staticmethod
    def _parse_json(
        text: str,
    ) -> dict:

        text = text.strip()

        # -----------------------------------------------------
        # Pure JSON
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Markdown fenced JSON
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Extract first JSON object
        # -----------------------------------------------------

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

        raise VLMClassificationError(
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

            raise VLMClassificationError(
                "Invalid classification: "
                f"{classification}"
            )

        # -----------------------------------------------------
        # Confidence
        # -----------------------------------------------------

        confidence = result.get(
            "confidence"
        )

        try:

            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError,
        ) as error:

            raise VLMClassificationError(
                "Confidence must be numeric."
            ) from error

        if not 0.0 <= confidence <= 1.0:

            raise VLMClassificationError(
                "Confidence must be between "
                "0.0 and 1.0."
            )

        # -----------------------------------------------------
        # Reason
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Signals
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Ad type
        # -----------------------------------------------------

        ad_type = result.get(
            "ad_type",
            "other",
        )

        if ad_type not in allowed_ad_types:
            ad_type = "other"

        # -----------------------------------------------------
        # Brand
        # -----------------------------------------------------

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

    def _build_output(
        self,
        video_path: Path,
        results: list,
        total_shots: int,
        vlm_calls: int,
        total_vlm_time: float,
    ) -> dict:

        counts = {
            "advertisement": 0,
            "organic_content": 0,
            "uncertain": 0,
        }

        skipped = 0

        for result in results:

            classification = result.get(
                "classification"
            )

            if classification in counts:

                counts[
                    classification
                ] += 1

            if result.get(
                "vlm_skipped",
                False,
            ):

                skipped += 1

        return {
            "schema_version": "1.1",

            "source": {
                "video": str(
                    video_path
                ),
            },

            "model": self.model,

            "frame_strategy": "middle",

            "vlm_strategy": (
                "evidence_gated_cpu"
            ),

            "shots": results,

            "stats": {
                "total_shots": total_shots,

                "vlm_calls": vlm_calls,

                "vlm_skipped": skipped,

                "advertisements": (
                    counts[
                        "advertisement"
                    ]
                ),

                "organic_content": (
                    counts[
                        "organic_content"
                    ]
                ),

                "uncertain": (
                    counts[
                        "uncertain"
                    ]
                ),

                "total_vlm_time_s": round(
                    total_vlm_time,
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

            raise VLMClassificationError(
                f"Could not read JSON: "
                f"{path}"
            ) from error

        except json.JSONDecodeError as error:

            raise VLMClassificationError(
                f"Invalid JSON: "
                f"{path}"
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
        vlm_calls: int,
        total_shots: int,
        total_vlm_time: float,
    ) -> None:

        counts = {
            "advertisement": 0,
            "organic_content": 0,
            "uncertain": 0,
        }

        skipped = 0

        for result in results:

            classification = result.get(
                "classification"
            )

            if classification in counts:

                counts[
                    classification
                ] += 1

            if result.get(
                "vlm_skipped",
                False,
            ):

                skipped += 1

        average_vlm_time = 0.0

        if vlm_calls:
            average_vlm_time = (
                total_vlm_time
                / vlm_calls
            )

        print()
        print("=" * 70)
        print("VLM SUMMARY")
        print("=" * 70)

        print(
            f"Total shots       : "
            f"{total_shots}"
        )

        print(
            f"VLM calls         : "
            f"{vlm_calls}"
        )

        print(
            f"VLM skipped       : "
            f"{skipped}"
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
            f"{total_vlm_time:.1f}s"
        )

        print(
            f"Average VLM call  : "
            f"{average_vlm_time:.1f}s"
        )

        print("=" * 70)


# =============================================================
# COMMAND LINE
# =============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "  python -m src.vlm_classifier "
            "<video_path>"
        )

        sys.exit(1)

    video_path = Path(
        sys.argv[1]
    )

    classifier = VLMClassifier()

    try:

        classifier.classify_video(
            video_path
        )

    except VLMClassificationError as error:

        print()
        print(
            "VLM CLASSIFICATION FAILED"
        )
        print(
            error
        )

        sys.exit(1)

    except KeyboardInterrupt:

        print()
        print(
            "VLM CLASSIFICATION INTERRUPTED"
        )

        sys.exit(130)


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    main()