from dataclasses import dataclass
from pathlib import Path

from src.probe import (
    MediaMetadata,
    MediaProbe,
    ProbeError,
)


class RoutingError(Exception):
    """Raised when media routing fails."""


@dataclass
class RoutingDecision:
    """
    Result of the media routing step.
    """

    media_path: str
    media_kind: str
    pipeline: str
    duration: float


class MediaRouter:
    """
    Route media to the appropriate detection pipeline.

    Current routes:

        long_form -> long_form pipeline
        short     -> short_form pipeline

    The router does NOT execute those pipelines yet.
    """

    def __init__(
        self,
        probe: MediaProbe | None = None,
    ):
        self.probe = probe or MediaProbe()

    # =========================================================
    # ROUTE MEDIA
    # =========================================================

    def route(
        self,
        media_path: str | Path,
    ) -> RoutingDecision:

        media_path = Path(media_path)

        try:
            metadata = self.probe.probe(
                media_path
            )

        except ProbeError as error:
            raise RoutingError(
                f"Could not probe media: {error}"
            ) from error

        pipeline = self._select_pipeline(
            metadata
        )

        return RoutingDecision(
            media_path=str(media_path),
            media_kind=metadata.media_kind,
            pipeline=pipeline,
            duration=metadata.duration,
        )

    # =========================================================
    # SELECT PIPELINE
    # =========================================================

    @staticmethod
    def _select_pipeline(
        metadata: MediaMetadata,
    ) -> str:

        if metadata.media_kind == "long_form":
            return "long_form"

        if metadata.media_kind == "short":
            return "short_form"

        raise RoutingError(
            f"Unknown media kind: "
            f"{metadata.media_kind}"
        )

    # =========================================================
    # DISPLAY DECISION
    # =========================================================

    @staticmethod
    def print_decision(
        decision: RoutingDecision,
    ) -> None:

        print()
        print("=" * 70)
        print("MEDIA ROUTING")
        print("=" * 70)

        print(
            f"Media       : "
            f"{decision.media_path}"
        )

        print(
            f"Duration    : "
            f"{decision.duration:.3f}s"
        )

        print(
            f"Media kind  : "
            f"{decision.media_kind}"
        )

        print(
            f"Pipeline    : "
            f"{decision.pipeline}"
        )

        print("=" * 70)


# =============================================================
# COMMAND-LINE TEST
# =============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "  python -m src.router "
            "<video_path>"
        )

        sys.exit(1)

    media_path = sys.argv[1]

    router = MediaRouter()

    try:

        decision = router.route(
            media_path
        )

        router.print_decision(
            decision
        )

    except RoutingError as error:

        print()
        print(
            "ROUTING FAILED"
        )

        print(error)

        sys.exit(1)