from pathlib import Path
import json
import sys


class ShotEvidenceError(Exception):
    """Raised when shot evidence construction fails."""


class ShotEvidenceBuilder:
    """
    Combine visual OCR evidence and audio ASR evidence
    into one shot-level evidence contract.

    Inputs:
        scenes.json
        shot_ocr.json
        shot_asr.json

    Output:
        shot_evidence.json
    """

    def __init__(
        self,
        minimum_ocr_confidence: float = 30.0,
    ):
        self.minimum_ocr_confidence = (
            minimum_ocr_confidence
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def build(
        self,
        scenes_path: str | Path,
        shot_ocr_path: str | Path,
        shot_asr_path: str | Path,
    ) -> dict:

        scenes_path = Path(scenes_path)
        shot_ocr_path = Path(shot_ocr_path)
        shot_asr_path = Path(shot_asr_path)

        scenes = self._load_json(
            scenes_path
        )

        shot_ocr = self._load_json(
            shot_ocr_path
        )

        shot_asr = self._load_json(
            shot_asr_path
        )

        shots = self._extract_shots(
            scenes
        )

        ocr_records = self._extract_records(
            shot_ocr,
            "shot_ocr",
        )

        asr_records = self._extract_records(
            shot_asr,
            "shot_asr",
        )

        ocr_by_shot = {
            self._shot_index(record): record
            for record in ocr_records
        }

        asr_by_shot = {
            self._shot_index(record): record
            for record in asr_records
        }

        evidence = []

        for position, shot in enumerate(
            shots,
            start=1,
        ):

            shot_index = self._get_shot_index(
                shot,
                position,
            )

            start = float(
                shot["start"]
            )

            end = float(
                shot["end"]
            )

            ocr_record = ocr_by_shot.get(
                shot_index,
                {},
            )

            asr_record = asr_by_shot.get(
                shot_index,
                {},
            )

            visual_evidence = (
                self._build_visual_evidence(
                    ocr_record
                )
            )

            audio_evidence = (
                self._build_audio_evidence(
                    asr_record
                )
            )

            evidence.append(
                {
                    "shot_index": shot_index,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(
                        end - start,
                        3,
                    ),
                    "visual": visual_evidence,
                    "audio": audio_evidence,
                }
            )

        return {
            "schema_version": "1.0",
            "shots": evidence,
            "stats": self._build_stats(
                evidence
            ),
        }

    # =========================================================
    # JSON LOADING
    # =========================================================

    @staticmethod
    def _load_json(
        path: Path,
    ):

        if not path.exists():
            raise ShotEvidenceError(
                f"File does not exist: {path}"
            )

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(file)

        except json.JSONDecodeError as error:

            raise ShotEvidenceError(
                f"Invalid JSON: {path}"
            ) from error

    # =========================================================
    # EXTRACT SHOTS
    # =========================================================

    @staticmethod
    def _extract_shots(
        scenes,
    ) -> list:

        if isinstance(
            scenes,
            dict,
        ):

            shots = scenes.get(
                "shots"
            )

        else:

            shots = None

        if not isinstance(
            shots,
            list,
        ):

            raise ShotEvidenceError(
                "scenes.json does not contain "
                "a valid 'shots' list."
            )

        return shots

    # =========================================================
    # EXTRACT RECORDS
    # =========================================================

    @staticmethod
    def _extract_records(
        data,
        name: str,
    ) -> list:

        if isinstance(
            data,
            list,
        ):
            return data

        if isinstance(
            data,
            dict,
        ):

            for key in (
                "shots",
                "results",
                "records",
            ):

                value = data.get(key)

                if isinstance(
                    value,
                    list,
                ):

                    return value

        raise ShotEvidenceError(
            f"{name}.json does not contain "
            "a valid list of records."
        )

    # =========================================================
    # SHOT INDEX
    # =========================================================

    @staticmethod
    def _shot_index(
        record: dict,
    ) -> int:

        value = record.get(
            "shot_index"
        )

        if value is None:

            value = record.get(
                "shot"
            )

        if value is None:

            raise ShotEvidenceError(
                "Evidence record is missing "
                "'shot_index'."
            )

        return int(value)

    @staticmethod
    def _get_shot_index(
        shot: dict,
        fallback: int,
    ) -> int:

        value = shot.get(
            "shot_index"
        )

        if value is None:

            value = shot.get(
                "index"
            )

        if value is None:

            return fallback

        return int(value)

    # =========================================================
    # VISUAL EVIDENCE
    # =========================================================

    def _build_visual_evidence(
        self,
        record: dict,
    ) -> dict:

        if not record:

            return {
                "ocr_text": "",
                "frames": [],
                "has_text": False,
            }

        frames = record.get(
            "frames"
        )

        if not isinstance(
            frames,
            list,
        ):

            frames = record.get(
                "results",
                [],
            )

        if not isinstance(
            frames,
            list,
        ):

            frames = []

        usable_frames = []

        text_values = []

        for frame in frames:

            if not isinstance(
                frame,
                dict,
            ):
                continue

            text = str(
                frame.get(
                    "text",
                    ""
                )
            ).strip()

            if (
                not text
                or text.lower()
                in {
                    "<unusable>",
                    "<none>",
                    "none",
                }
            ):
                continue

            confidence = frame.get(
                "confidence"
            )

            try:

                confidence = float(
                    confidence
                )

            except (
                TypeError,
                ValueError,
            ):

                confidence = 0.0

            if (
                confidence
                < self.minimum_ocr_confidence
            ):
                continue

            usable_frames.append(
                {
                    "position": frame.get(
                        "position"
                    ),
                    "timestamp": frame.get(
                        "timestamp"
                    ),
                    "text": text,
                    "confidence": round(
                        confidence,
                        2,
                    ),
                }
            )

            text_values.append(
                text
            )

        combined_text = self._deduplicate_text(
            text_values
        )

        return {
            "ocr_text": combined_text,
            "frames": usable_frames,
            "has_text": bool(
                combined_text
            ),
        }

    # =========================================================
    # AUDIO EVIDENCE
    # =========================================================

    @staticmethod
    def _build_audio_evidence(
        record: dict,
    ) -> dict:

        if not record:

            return {
                "transcript": "",
                "segments": [],
                "has_speech": False,
            }

        segments = record.get(
            "transcript"
        )

        if not isinstance(
            segments,
            list,
        ):

            segments = []

        clean_segments = []

        text_values = []

        for segment in segments:

            if not isinstance(
                segment,
                dict,
            ):
                continue

            text = str(
                segment.get(
                    "text",
                    ""
                )
            ).strip()

            if not text:
                continue

            clean_segments.append(
                {
                    "start": segment.get(
                        "start"
                    ),
                    "end": segment.get(
                        "end"
                    ),
                    "overlap_start": segment.get(
                        "overlap_start"
                    ),
                    "overlap_end": segment.get(
                        "overlap_end"
                    ),
                    "overlap_duration": segment.get(
                        "overlap_duration"
                    ),
                    "text": text,
                }
            )

            text_values.append(
                text
            )

        combined_text = " ".join(
            text_values
        ).strip()

        return {
            "transcript": combined_text,
            "segments": clean_segments,
            "has_speech": bool(
                combined_text
            ),
        }

    # =========================================================
    # TEXT DEDUPLICATION
    # =========================================================

    @staticmethod
    def _deduplicate_text(
        values: list[str],
    ) -> str:

        result = []

        seen = set()

        for value in values:

            normalized = " ".join(
                value.split()
            ).lower()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            result.append(
                value
            )

        return " ".join(
            result
        ).strip()

    # =========================================================
    # STATISTICS
    # =========================================================

    @staticmethod
    def _build_stats(
        evidence: list,
    ) -> dict:

        total = len(
            evidence
        )

        with_ocr = sum(
            1
            for item in evidence
            if item["visual"]["has_text"]
        )

        with_speech = sum(
            1
            for item in evidence
            if item["audio"]["has_speech"]
        )

        with_both = sum(
            1
            for item in evidence
            if (
                item["visual"]["has_text"]
                and item["audio"]["has_speech"]
            )
        )

        return {
            "total_shots": total,
            "shots_with_ocr": with_ocr,
            "shots_with_speech": with_speech,
            "shots_with_both": with_both,
            "shots_without_evidence": (
                total
                - sum(
                    1
                    for item in evidence
                    if (
                        item["visual"]["has_text"]
                        or item["audio"]["has_speech"]
                    )
                )
            ),
        }

    # =========================================================
    # DISPLAY
    # =========================================================

    @staticmethod
    def print_summary(
        result: dict,
    ) -> None:

        shots = result.get(
            "shots",
            [],
        )

        stats = result.get(
            "stats",
            {},
        )

        print()
        print("=" * 70)
        print("SHOT EVIDENCE")
        print("=" * 70)

        print(
            f"Shots              : "
            f"{stats.get('total_shots', 0)}"
        )

        print(
            f"Shots with OCR     : "
            f"{stats.get('shots_with_ocr', 0)}"
        )

        print(
            f"Shots with speech  : "
            f"{stats.get('shots_with_speech', 0)}"
        )

        print(
            f"Shots with both    : "
            f"{stats.get('shots_with_both', 0)}"
        )

        print(
            f"No evidence       : "
            f"{stats.get('shots_without_evidence', 0)}"
        )

        print()
        print("PER-SHOT EVIDENCE")
        print("-" * 70)

        for shot in shots:

            print(
                f"Shot {shot['shot_index']:02d}: "
                f"{shot['start']:.3f}s → "
                f"{shot['end']:.3f}s"
            )

            visual = shot[
                "visual"
            ]

            audio = shot[
                "audio"
            ]

            if visual["has_text"]:

                print(
                    f"  OCR : "
                    f"{visual['ocr_text']}"
                )

            else:

                print(
                    "  OCR : <none>"
                )

            if audio["has_speech"]:

                print(
                    f"  ASR : "
                    f"{audio['transcript']}"
                )

            else:

                print(
                    "  ASR : <none>"
                )

        print("=" * 70)

    # =========================================================
    # SAVE
    # =========================================================

    @staticmethod
    def save(
        result: dict,
        output_path: str | Path,
    ) -> Path:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
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

        print()
        print(
            f"Shot evidence saved: "
            f"{output_path}"
        )

        return output_path


# =============================================================
# COMMAND LINE
# =============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python -m src.shot_evidence "
            "<video_path>"
        )

        sys.exit(1)

    video_path = Path(
        sys.argv[1]
    )

    video_name = video_path.stem

    base_dir = (
        Path("data/outputs")
        / video_name
    )

    scenes_path = (
        base_dir
        / "scenes.json"
    )

    shot_ocr_path = (
        base_dir
        / "shot_ocr.json"
    )

    shot_asr_path = (
        base_dir
        / "shot_asr.json"
    )

    output_path = (
        base_dir
        / "shot_evidence.json"
    )

    builder = ShotEvidenceBuilder()

    try:

        result = builder.build(
            scenes_path=scenes_path,
            shot_ocr_path=shot_ocr_path,
            shot_asr_path=shot_asr_path,
        )

        builder.print_summary(
            result
        )

        builder.save(
            result,
            output_path,
        )

    except ShotEvidenceError as error:

        print()
        print(
            "SHOT EVIDENCE FAILED"
        )

        print(error)

        sys.exit(1)