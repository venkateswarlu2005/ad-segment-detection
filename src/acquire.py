from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlparse


class AcquisitionError(Exception):
    """Raised when media acquisition fails."""


class MediaAcquirer:
    """
    Acquire media from either:

    1. A local video file
    2. A URL supported by yt-dlp

    All acquired videos are stored under data/videos/.
    """

    def __init__(
        self,
        output_dir: str | Path = "data/videos",
    ):
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ---------------------------------------------------------

    def acquire(
        self,
        source: str | Path,
        output_name: str | None = None,
    ) -> Path:
        """
        Acquire a video from a local path or URL.

        Returns
        -------
        Path
            Path to the normalized local video.
        """

        source = str(source)

        if self._is_url(source):
            return self._acquire_url(
                source,
                output_name,
            )

        return self._acquire_local(
            Path(source),
            output_name,
        )

    # ---------------------------------------------------------
    # DETECT URL
    # ---------------------------------------------------------

    @staticmethod
    def _is_url(source: str) -> bool:
        """
        Determine whether a source is an HTTP/HTTPS URL.
        """

        try:
            parsed = urlparse(source)

            return parsed.scheme in {
                "http",
                "https",
            }

        except Exception:
            return False

    # ---------------------------------------------------------
    # LOCAL FILE
    # ---------------------------------------------------------

    def _acquire_local(
        self,
        source_path: Path,
        output_name: str | None = None,
    ) -> Path:
        """
        Copy a local video into data/videos/.
        """

        if not source_path.exists():
            raise AcquisitionError(
                f"Local file does not exist: "
                f"{source_path}"
            )

        if not source_path.is_file():
            raise AcquisitionError(
                f"Source is not a file: "
                f"{source_path}"
            )

        extension = source_path.suffix.lower()

        supported_extensions = {
            ".mp4",
            ".mov",
            ".mkv",
            ".webm",
            ".avi",
            ".m4v",
        }

        if extension not in supported_extensions:
            raise AcquisitionError(
                f"Unsupported video extension: "
                f"{extension}"
            )

        if output_name:
            destination = (
                self.output_dir
                / output_name
            )

            if destination.suffix == "":
                destination = (
                    destination
                    .with_suffix(extension)
                )
        else:
            destination = (
                self.output_dir
                / source_path.name
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Avoid copying a file onto itself.
        try:
            same_file = (
                source_path.resolve()
                == destination.resolve()
            )
        except FileNotFoundError:
            same_file = False

        if not same_file:
            shutil.copy2(
                source_path,
                destination,
            )

        print()
        print("=" * 70)
        print("MEDIA ACQUISITION")
        print("=" * 70)
        print("Source type : local file")
        print(f"Source      : {source_path}")
        print(f"Destination : {destination}")
        print("=" * 70)

        return destination

    # ---------------------------------------------------------
    # URL
    # ---------------------------------------------------------

    def _acquire_url(
        self,
        url: str,
        output_name: str | None = None,
    ) -> Path:
        """
        Download a video URL using yt-dlp.
        """

        print()
        print("=" * 70)
        print("MEDIA ACQUISITION")
        print("=" * 70)
        print("Source type : URL")
        print(f"URL         : {url}")
        print("=" * 70)

        if output_name:
            output_template = (
                self.output_dir
                / output_name
            )

            # yt-dlp adds the extension automatically.
            output_template = str(
                output_template
            )
        else:
            output_template = str(
                self.output_dir
                / "%(id)s.%(ext)s"
            )

        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--restrict-filenames",
            "-f",
            "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
            "--merge-output-format",
            "mp4",
            "-o",
            output_template,
            url,
        ]

        print("Downloading with yt-dlp...")

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise AcquisitionError(
                "yt-dlp failed.\n\n"
                + result.stderr
            )

        # -----------------------------------------------------
        # FIND RESULT
        # -----------------------------------------------------

        if output_name:
            expected = (
                self.output_dir
                / output_name
            )

            candidates = []

            if expected.exists():
                candidates.append(expected)

            candidates.extend(
                self.output_dir.glob(
                    f"{output_name}.*"
                )
            )

        else:
            # Find recently created video files.
            candidates = []

            for extension in [
                ".mp4",
                ".mkv",
                ".webm",
                ".mov",
            ]:
                candidates.extend(
                    self.output_dir.glob(
                        f"*{extension}"
                    )
                )

            candidates.sort(
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )

        if not candidates:
            raise AcquisitionError(
                "yt-dlp completed successfully, "
                "but no downloaded video was found "
                f"in {self.output_dir}"
            )

        downloaded_file = candidates[0]

        print()
        print(
            f"Downloaded video: "
            f"{downloaded_file}"
        )

        print("=" * 70)

        return downloaded_file


# -------------------------------------------------------------
# COMMAND-LINE TEST
# -------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage:"
        )
        print(
            "  python -m src.acquire "
            "<local_file_or_url>"
        )
        sys.exit(1)

    source = sys.argv[1]

    acquirer = MediaAcquirer()

    try:
        result = acquirer.acquire(source)

        print()
        print(
            "ACQUISITION SUCCESSFUL"
        )
        print(
            f"Local video: {result}"
        )

    except AcquisitionError as error:
        print()
        print(
            "ACQUISITION FAILED"
        )
        print(error)
        sys.exit(1)