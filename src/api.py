import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.acquire import (
    AcquisitionError,
    MediaAcquirer,
)

from src.pipeline import (
    MediaPipeline,
    PipelineError,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Ad Segment Detection API",
    description=(
        "API for acquiring videos and detecting "
        "advertisement segments."
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST MODELS
# ============================================================


class AcquireRequest(BaseModel):
    source: str = Field(
        ...,
        description="Video URL or local file path",
        min_length=1,
    )

    output_name: str | None = Field(
        default=None,
        description="Optional local filename",
    )


class ProcessRequest(BaseModel):
    source: str = Field(
        ...,
        description="Video URL or local file path",
        min_length=1,
    )

    output_name: str | None = Field(
        default=None,
        description="Optional local filename",
    )


# ============================================================
# RESPONSE MODELS
# ============================================================


class AcquireResponse(BaseModel):
    success: bool
    source_url: str
    local_path: str | None = None
    filename: str | None = None
    error: str | None = None


class ProcessResponse(BaseModel):
    success: bool
    source: str
    video_path: str | None = None
    media_kind: str | None = None
    segments: list = Field(default_factory=list)
    result_path: str | None = None
    error: str | None = None


# ============================================================
# SERVICES
# ============================================================


acquirer = MediaAcquirer(
    output_dir="data/videos"
)

pipeline = MediaPipeline()


# ============================================================
# ROOT
# ============================================================


@app.get("/")
def root():
    return {
        "service": "Ad Segment Detection API",
        "status": "running",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH
# ============================================================


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


# ============================================================
# ACQUIRE VIDEO
# ============================================================


@app.post(
    "/acquire",
    response_model=AcquireResponse,
)
def acquire_video(
    request: AcquireRequest,
):
    source = request.source.strip()

    if not source:
        raise HTTPException(
            status_code=400,
            detail="Source cannot be empty.",
        )

    try:
        video_path = acquirer.acquire(
            source=source,
            output_name=request.output_name,
        )

        video_path = Path(video_path)

        return AcquireResponse(
            success=True,
            source_url=source,
            local_path=str(video_path),
            filename=video_path.name,
        )

    except AcquisitionError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected acquisition error: "
                f"{error}"
            ),
        ) from error


# ============================================================
# PROCESS VIDEO
# ============================================================


@app.post(
    "/process",
    response_model=ProcessResponse,
)
def process_video(
    request: ProcessRequest,
):
    source = request.source.strip()

    if not source:
        raise HTTPException(
            status_code=400,
            detail="Source cannot be empty.",
        )

    # --------------------------------------------------------
    # STEP 1: ACQUIRE VIDEO
    # --------------------------------------------------------

    try:
        video_path = acquirer.acquire(
            source=source,
            output_name=request.output_name,
        )

        video_path = Path(video_path)

    except AcquisitionError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected acquisition error: "
                f"{error}"
            ),
        ) from error

    # --------------------------------------------------------
    # STEP 2: RUN COMPLETE PIPELINE
    # --------------------------------------------------------

    try:
        result = pipeline.process(
            video_path
        )

    except PipelineError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Video processing failed: "
                f"{error}"
            ),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected processing error: "
                f"{error}"
            ),
        ) from error

    # --------------------------------------------------------
    # STEP 3: EXTRACT RESULT
    # --------------------------------------------------------

    output_dir = (
        Path("data")
        / "outputs"
        / video_path.stem
    )

    result_path = (
        output_dir
        / "result.json"
    )

    source_data = result.get(
        "source",
        {},
    )

    media_kind = (
        source_data.get("kind")
        or source_data.get("media_kind")
    )

    segments = result.get(
        "segments",
        [],
    )

    # --------------------------------------------------------
    # STEP 4: RETURN RESPONSE
    # --------------------------------------------------------

    return ProcessResponse(
        success=True,
        source=source,
        video_path=str(video_path),
        media_kind=media_kind,
        segments=segments,
        result_path=str(result_path),
    )


# ============================================================
# GET RESULT
# ============================================================


@app.get(
    "/result/{video_name}"
)
def get_result(
    video_name: str,
):
    safe_name = Path(video_name).stem

    if not safe_name:
        raise HTTPException(
            status_code=400,
            detail="Invalid video name.",
        )

    result_path = (
        Path("data")
        / "outputs"
        / safe_name
        / "result.json"
    )

    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No result found for "
                f"'{safe_name}'."
            ),
        )

    try:
        with result_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            result = json.load(file)

        return result

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Result file contains "
                "invalid JSON."
            ),
        ) from error

    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not read result: "
                f"{error}"
            ),
        ) from error


# ============================================================
# GET STATUS
# ============================================================


@app.get(
    "/status/{video_name}"
)
def get_status(
    video_name: str,
):
    safe_name = Path(video_name).stem

    if not safe_name:
        raise HTTPException(
            status_code=400,
            detail="Invalid video name.",
        )

    output_dir = (
        Path("data")
        / "outputs"
        / safe_name
    )

    # --------------------------------------------------------
    # VIDEO DOES NOT EXIST
    # --------------------------------------------------------

    if not output_dir.exists():
        return {
            "video": safe_name,
            "status": "not_found",
        }

    # --------------------------------------------------------
    # FINAL RESULT EXISTS
    # --------------------------------------------------------

    result_path = (
        output_dir
        / "result.json"
    )

    if result_path.exists():

        try:
            with result_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                result = json.load(file)

            source_data = result.get(
                "source",
                {},
            )

            stats = result.get(
                "stats",
                {},
            )

            return {
                "video": safe_name,
                "status": "completed",
                "media_kind": source_data.get(
                    "kind"
                ),
                "segments": len(
                    result.get(
                        "segments",
                        [],
                    )
                ),
                "advertisement_duration": (
                    stats.get(
                        "advertisement_duration",
                        0,
                    )
                ),
                "result_path": str(
                    result_path
                ),
            }

        except Exception as error:
            return {
                "video": safe_name,
                "status": "result_error",
                "error": str(error),
            }

    # --------------------------------------------------------
    # CHECK PIPELINE PROGRESS
    # --------------------------------------------------------

    completed_steps = []

    step_files = {
        "probe": "metadata.json",
        "asr": "transcript.json",
        "long_scene": "long_scenes.json",
        "long_evidence": "long_evidence.json",
        "long_candidate_detector": (
            "long_candidates.json"
        ),
        "long_vlm_classifier": (
            "long_vlm_results.json"
        ),
        "long_segment_merger": (
            "long_segments.json"
        ),
    }

    for step, filename in step_files.items():

        file_path = (
            output_dir
            / filename
        )

        if file_path.exists():
            completed_steps.append(
                step
            )

    # --------------------------------------------------------
    # RETURN CURRENT STATUS
    # --------------------------------------------------------

    if completed_steps:
        return {
            "video": safe_name,
            "status": "processing_or_incomplete",
            "completed_steps": completed_steps,
            "output_dir": str(
                output_dir
            ),
        }

    return {
        "video": safe_name,
        "status": "acquired",
        "output_dir": str(
            output_dir
        ),
    }


# ============================================================
# COMMAND LINE
# ============================================================


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )