from pathlib import Path
import subprocess


class FrameExtractor:
    """
    Extract video frames at a fixed sampling rate.
    """

    def __init__(
        self,
        output_dir: str | Path = "data/frames",
    ):
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def extract(
        self,
        video_path: str | Path,
        fps: float = 1.0,
    ) -> list[Path]:
        """
        Extract frames from a video.

        Parameters
        ----------
        video_path:
            Path to the input video.

        fps:
            Number of frames to extract per second.

        Returns
        -------
        list[Path]
            Paths to extracted frames.
        """

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video file not found: {video_path}"
            )

        if fps <= 0:
            raise ValueError(
                "FPS must be greater than zero."
            )

        video_output_dir = (
            self.output_dir / video_path.stem
        )

        video_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_pattern = (
            video_output_dir / "frame_%06d.jpg"
        )

        print()
        print("=" * 70)
        print("FRAME EXTRACTION")
        print("=" * 70)
        print(f"Video : {video_path}")
        print(f"FPS   : {fps}")
        print(f"Output: {video_output_dir}")
        print("=" * 70)

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            "-q:v",
            "2",
            str(output_pattern),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg frame extraction failed.\n\n"
                + result.stderr
            )

        frames = sorted(
            video_output_dir.glob("frame_*.jpg")
        )

        if not frames:
            raise RuntimeError(
                "FFmpeg completed but no frames "
                "were created."
            )

        print()
        print(
            f"Extracted {len(frames)} frames."
        )

        print("=" * 70)

        return frames


if __name__ == "__main__":

    extractor = FrameExtractor()

    frames = extractor.extract(
        "data/videos/audio_test.mp4",
        fps=1.0,
    )

    print()
    print("First few frames:")

    for frame in frames[:5]:
        print(frame)

    print()
    print(
        f"Total frames: {len(frames)}"
    )