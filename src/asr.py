from dataclasses import asdict, dataclass
from pathlib import Path
import json
import shutil
import subprocess
import sys

import whisper


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class ASRProcessor:
    """
    Handles the complete ASR workflow:

    1. Load Whisper
    2. Extract audio from a video when required
    3. Transcribe audio
    4. Save timestamped transcript as JSON
    5. Save human-readable transcript as TXT
    """

    def __init__(
        self,
        model_name: str = "tiny",
        audio_dir: str | Path = "data/audio",
        output_dir: str | Path = "data/outputs",
    ):
        self.model_name = model_name

        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)

        self.audio_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(f"Loading Whisper model: {model_name}")

        self.model = whisper.load_model(model_name)

        print("Whisper model loaded successfully.")

    # ---------------------------------------------------------
    # AUDIO EXTRACTION
    # ---------------------------------------------------------

    def extract_audio(
        self,
        video_path: str | Path,
    ) -> Path:
        """
        Extract audio from a video.

        Output:
            16 kHz
            mono
            PCM WAV
        """

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found: {video_path}"
            )

        output_audio = (
            self.audio_dir
            / f"{video_path.stem}.wav"
        )

        print()
        print("Extracting audio...")
        print(f"Input : {video_path}")
        print(f"Output: {output_audio}")

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_audio),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg audio extraction failed.\n\n"
                + result.stderr
            )

        if not output_audio.exists():
            raise RuntimeError(
                "FFmpeg completed but the audio file "
                "was not created."
            )

        print("Audio extraction complete.")

        return output_audio

    # ---------------------------------------------------------
    # TRANSCRIPTION
    # ---------------------------------------------------------

    def transcribe(
        self,
        audio_path: str | Path,
    ) -> list[TranscriptSegment]:
        """
        Transcribe an audio file with Whisper.
        """

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        print()
        print("Transcribing audio...")
        print(f"Audio: {audio_path}")

        result = self.model.transcribe(
            str(audio_path),
            fp16=False,
            verbose=False,
        )

        segments: list[TranscriptSegment] = []

        for segment in result.get(
            "segments",
            [],
        ):
            text = segment.get(
                "text",
                "",
            ).strip()

            if not text:
                continue

            segments.append(
                TranscriptSegment(
                    start=float(
                        segment["start"]
                    ),
                    end=float(
                        segment["end"]
                    ),
                    text=text,
                )
            )

        print(
            f"Transcription complete. "
            f"Segments: {len(segments)}"
        )

        return segments

    # ---------------------------------------------------------
    # SAVE JSON
    # ---------------------------------------------------------

    def save_transcript_json(
        self,
        segments: list[TranscriptSegment],
        output_path: str | Path,
        source_video: str | Path | None = None,
        source_audio: str | Path | None = None,
    ) -> Path:
        """
        Save timestamped transcript as JSON.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "source": {
                "video": (
                    str(source_video)
                    if source_video
                    else None
                ),
                "audio": (
                    str(source_audio)
                    if source_audio
                    else None
                ),
            },
            "model": self.model_name,
            "segments": [
                asdict(segment)
                for segment in segments
            ],
        }

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
            f"Transcript JSON saved: "
            f"{output_path}"
        )

        return output_path

    # ---------------------------------------------------------
    # SAVE TXT
    # ---------------------------------------------------------

    def save_transcript_txt(
        self,
        segments: list[TranscriptSegment],
        output_path: str | Path,
    ) -> Path:
        """
        Save a human-readable transcript.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            for segment in segments:
                file.write(
                    f"[{segment.start:.2f}s - "
                    f"{segment.end:.2f}s] "
                    f"{segment.text}\n"
                )

        print(
            f"Transcript TXT saved: "
            f"{output_path}"
        )

        return output_path

    # ---------------------------------------------------------
    # PRESERVE VIDEO
    # ---------------------------------------------------------

    def preserve_video(
        self,
        video_path: str | Path,
        output_directory: str | Path,
    ) -> Path:
        """
        Copy the source video into the output directory.

        We do not modify the original video.
        """

        video_path = Path(video_path)
        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = (
            output_directory
            / video_path.name
        )

        if video_path.resolve() != destination.resolve():
            shutil.copy2(
                video_path,
                destination,
            )

        print(
            f"Video preserved: {destination}"
        )

        return destination

    # ---------------------------------------------------------
    # COMPLETE VIDEO WORKFLOW
    # ---------------------------------------------------------

    def process_video(
        self,
        video_path: str | Path,
    ) -> dict:
        """
        Complete video → audio → transcript workflow.

        Returns a dictionary containing all generated paths
        and transcript segments.
        """

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found: {video_path}"
            )

        run_directory = (
            self.output_dir
            / video_path.stem
        )

        run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print("=" * 70)
        print("ASR PIPELINE")
        print("=" * 70)
        print(f"Video: {video_path}")
        print("=" * 70)

        # 1. Preserve/copy video
        saved_video = self.preserve_video(
            video_path,
            run_directory,
        )

        # 2. Extract audio
        audio_path = self.extract_audio(
            video_path
        )

        # 3. Transcribe
        segments = self.transcribe(
            audio_path
        )

        # 4. Save JSON transcript
        transcript_json = (
            run_directory
            / "transcript.json"
        )

        self.save_transcript_json(
            segments=segments,
            output_path=transcript_json,
            source_video=saved_video,
            source_audio=audio_path,
        )

        # 5. Save readable TXT transcript
        transcript_txt = (
            run_directory
            / "transcript.txt"
        )

        self.save_transcript_txt(
            segments=segments,
            output_path=transcript_txt,
        )

        print()
        print("=" * 70)
        print("ASR PIPELINE COMPLETE")
        print("=" * 70)

        print(
            f"Video      : {saved_video}"
        )

        print(
            f"Audio      : {audio_path}"
        )

        print(
            f"Transcript : {transcript_json}"
        )

        print(
            f"Readable   : {transcript_txt}"
        )

        print(
            f"Segments   : {len(segments)}"
        )

        print("=" * 70)

        return {
            "video": saved_video,
            "audio": audio_path,
            "transcript_json": transcript_json,
            "transcript_txt": transcript_txt,
            "segments": segments,
        }


# -------------------------------------------------------------
# DIRECT TEST
# -------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.asr <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    processor = ASRProcessor()

    try:
        processor.process_video(video_path)

    except FileNotFoundError as error:
        print()
        print("ASR FAILED")
        print(error)
        sys.exit(1)

    except Exception as error:
        print()
        print("ASR FAILED")
        print(error)
        sys.exit(1)