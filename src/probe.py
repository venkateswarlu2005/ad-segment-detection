from dataclasses import asdict, dataclass
from pathlib import Path
import json
import subprocess
from fractions import Fraction


class ProbeError(Exception):
    """Raised when media metadata probing fails."""


@dataclass
class VideoStreamInfo:
    codec_name: str | None
    width: int | None
    height: int | None
    fps: float | None


@dataclass
class AudioStreamInfo:
    codec_name: str | None
    sample_rate: int | None
    channels: int | None


@dataclass
class MediaMetadata:
    path: str
    filename: str
    duration: float
    media_kind: str
    video: VideoStreamInfo | None
    audio: AudioStreamInfo | None


class MediaProbe:
    """
    Probe local media files using FFprobe.

    Responsibilities:
        1. Validate that the file exists.
        2. Extract media metadata.
        3. Determine whether the media is
           long-form or short/reel.

    This class does NOT perform ASR, OCR,
    VLM analysis, or advertisement detection.
    """

    def __init__(
        self,
        long_form_threshold: float = 120.0,
    ):
        """
        Parameters
        ----------
        long_form_threshold:
            Duration in seconds above which media
            is initially classified as long-form.

        120 seconds is a temporary routing threshold.
        We can change it later based on the assignment's
        actual evaluation requirements.
        """

        self.long_form_threshold = (
            long_form_threshold
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def probe(
        self,
        media_path: str | Path,
    ) -> MediaMetadata:

        media_path = Path(media_path)

        self._validate_file(
            media_path
        )

        raw_metadata = (
            self._run_ffprobe(media_path)
        )

        video_info = (
            self._extract_video_stream(
                raw_metadata
            )
        )

        audio_info = (
            self._extract_audio_stream(
                raw_metadata
            )
        )

        duration = (
            self._extract_duration(
                raw_metadata
            )
        )

        media_kind = (
            self.classify_media(
                duration
            )
        )

        return MediaMetadata(
            path=str(media_path),
            filename=media_path.name,
            duration=duration,
            media_kind=media_kind,
            video=video_info,
            audio=audio_info,
        )

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_file(
        media_path: Path,
    ) -> None:

        if not media_path.exists():
            raise ProbeError(
                f"Media file does not exist: "
                f"{media_path}"
            )

        if not media_path.is_file():
            raise ProbeError(
                f"Media path is not a file: "
                f"{media_path}"
            )

    # =========================================================
    # FFPROBE
    # =========================================================

    @staticmethod
    def _run_ffprobe(
        media_path: Path,
    ) -> dict:

        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration:"
                "stream="
                "codec_type,"
                "codec_name,"
                "width,"
                "height,"
                "r_frame_rate,"
                "sample_rate,"
                "channels"
            ),
            "-of",
            "json",
            str(media_path),
        ]

        try:

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        except FileNotFoundError as error:

            raise ProbeError(
                "ffprobe was not found. "
                "Make sure FFmpeg is installed "
                "and available in PATH."
            ) from error

        if result.returncode != 0:

            raise ProbeError(
                "ffprobe failed:\n"
                + result.stderr
            )

        try:

            return json.loads(
                result.stdout
            )

        except json.JSONDecodeError as error:

            raise ProbeError(
                "Could not parse ffprobe output."
            ) from error

    # =========================================================
    # VIDEO STREAM
    # =========================================================

    @staticmethod
    def _extract_video_stream(
        metadata: dict,
    ) -> VideoStreamInfo | None:

        streams = metadata.get(
            "streams",
            [],
        )

        for stream in streams:

            if stream.get(
                "codec_type"
            ) != "video":
                continue

            fps = MediaProbe._parse_fps(
                stream.get(
                    "r_frame_rate"
                )
            )

            return VideoStreamInfo(
                codec_name=stream.get(
                    "codec_name"
                ),
                width=stream.get(
                    "width"
                ),
                height=stream.get(
                    "height"
                ),
                fps=fps,
            )

        return None

    # =========================================================
    # AUDIO STREAM
    # =========================================================

    @staticmethod
    def _extract_audio_stream(
        metadata: dict,
    ) -> AudioStreamInfo | None:

        streams = metadata.get(
            "streams",
            [],
        )

        for stream in streams:

            if stream.get(
                "codec_type"
            ) != "audio":
                continue

            sample_rate = (
                stream.get(
                    "sample_rate"
                )
            )

            if sample_rate is not None:
                sample_rate = int(
                    sample_rate
                )

            channels = (
                stream.get(
                    "channels"
                )
            )

            if channels is not None:
                channels = int(
                    channels
                )

            return AudioStreamInfo(
                codec_name=stream.get(
                    "codec_name"
                ),
                sample_rate=sample_rate,
                channels=channels,
            )

        return None

    # =========================================================
    # DURATION
    # =========================================================

    @staticmethod
    def _extract_duration(
        metadata: dict,
    ) -> float:

        format_data = metadata.get(
            "format",
            {},
        )

        raw_duration = (
            format_data.get(
                "duration"
            )
        )

        if raw_duration is None:
            raise ProbeError(
                "Media duration could not be determined."
            )

        try:

            duration = float(
                raw_duration
            )

        except (
            ValueError,
            TypeError,
        ) as error:

            raise ProbeError(
                f"Invalid duration: "
                f"{raw_duration}"
            ) from error

        if duration < 0:
            raise ProbeError(
                f"Invalid negative duration: "
                f"{duration}"
            )

        return duration

    # =========================================================
    # FPS
    # =========================================================

    @staticmethod
    def _parse_fps(
        value: str | None,
    ) -> float | None:

        if not value:
            return None

        try:

            fps = float(
                Fraction(value)
            )

            if fps <= 0:
                return None

            return round(
                fps,
                3,
            )

        except (
            ValueError,
            ZeroDivisionError,
        ):
            return None

    # =========================================================
    # MEDIA CLASSIFICATION
    # =========================================================

    def classify_media(
        self,
        duration: float,
    ) -> str:

        if duration <= 0:
            raise ValueError(
                "Duration must be greater than zero."
            )

        if (
            duration
            >= self.long_form_threshold
        ):
            return "long_form"

        return "short"

    # =========================================================
    # DISPLAY
    # =========================================================

    @staticmethod
    def print_metadata(
        metadata: MediaMetadata,
    ) -> None:

        print()
        print("=" * 70)
        print("MEDIA METADATA")
        print("=" * 70)

        print(
            f"File       : "
            f"{metadata.filename}"
        )

        print(
            f"Path       : "
            f"{metadata.path}"
        )

        print(
            f"Duration   : "
            f"{metadata.duration:.3f}s"
        )

        print(
            f"Media kind : "
            f"{metadata.media_kind}"
        )

        print()

        if metadata.video:

            print("VIDEO")

            print(
                f"  Codec    : "
                f"{metadata.video.codec_name}"
            )

            print(
                f"  Resolution: "
                f"{metadata.video.width}x"
                f"{metadata.video.height}"
            )

            print(
                f"  FPS      : "
                f"{metadata.video.fps}"
            )

        else:

            print("VIDEO: none")

        print()

        if metadata.audio:

            print("AUDIO")

            print(
                f"  Codec    : "
                f"{metadata.audio.codec_name}"
            )

            print(
                f"  Sample rate: "
                f"{metadata.audio.sample_rate}"
            )

            print(
                f"  Channels : "
                f"{metadata.audio.channels}"
            )

        else:

            print("AUDIO: none")

        print("=" * 70)

    # =========================================================
    # JSON
    # =========================================================

    @staticmethod
    def save_metadata(
        metadata: MediaMetadata,
        output_path: str | Path,
    ) -> Path:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = asdict(
            metadata
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"Metadata saved: "
            f"{output_path}"
        )

        return output_path


# =============================================================
# COMMAND-LINE TEST
# =============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "  python -m src.probe "
            "<video_path>"
        )

        sys.exit(1)

    media_path = sys.argv[1]

    probe = MediaProbe(
        long_form_threshold=120.0
    )

    try:

        metadata = probe.probe(
            media_path
        )

        probe.print_metadata(
            metadata
        )

        output_path = (
            Path("data/outputs")
            / Path(media_path).stem
            / "metadata.json"
        )

        probe.save_metadata(
            metadata,
            output_path,
        )

    except (
        ProbeError,
        ValueError,
    ) as error:

        print()
        print(
            "PROBE FAILED"
        )
        print(error)

        sys.exit(1)