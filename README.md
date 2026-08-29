# Ad Segment Detection in Video

An end-to-end video advertisement detection system that accepts a video URL or local video file and produces a machine-readable timeline of detected advertising segments.

The system supports:

- YouTube long-form videos
- YouTube Shorts / short vertical videos
- Local video files
- Long-form sponsor reads
- Visual advertisements
- Promotional content
- OCR and speech-based signals
- ASR using Whisper
- VLM-based commercial verification
- FastAPI HTTP API
- Swagger UI
- Automatic short-form / long-form routing
- Final machine-readable JSON output

---

## 1. Problem

Given a video, the system identifies segments that are likely to contain advertising or commercial content.

For each detected segment, the system attempts to provide:

- Start timestamp
- End timestamp
- Advertisement type
- Confidence
- Brand, when identifiable
- Description/reason
- Evidence used for the decision

The system combines multiple signals rather than relying on a single classifier.

---

# 2. High-Level Architecture

```text
                         VIDEO URL / FILE
                                |
                                v
                      +-------------------+
                      |   Media Acquire   |
                      |      yt-dlp       |
                      +---------+---------+
                                |
                                v
                         +-------------+
                         |    Probe    |
                         |  metadata   |
                         +------+------+ 
                                |
                    +-----------+-----------+
                    |                       |
                 SHORT                    LONG
                    |                       |
                    v                       v
             Scene Detection             ASR
                    |                       |
             Representative Frames    Long Scene Detection
                    |                       |
                   OCR               Long Evidence Extraction
                    |                       |
                   ASR               Candidate Detection
                    |                       |
             Shot Alignment          Long VLM Verification
                    |                       |
             Shot Evidence            Segment Merging
                    |                       |
                  VLM                       |
                    |                       |
             Advertisement                 |
                Scoring                    |
                    |                       |
             Segment Merging                |
                    |                       |
                    +-----------+-----------+
                                |
                                v
                         Final Contract
                                |
                                v
                           result.json
```

---

# 3. Why Two Pipelines?

Short-form and long-form videos have different advertisement characteristics.

## Short-form

Short videos often contain:

- Visual promotional content
- Text overlays
- Product imagery
- Rapid scene changes
- Short advertisements

The short-form pipeline therefore emphasizes shot-level visual and audio evidence.

```text
Scenes
  ↓
Representative Frames
  ↓
OCR
  ↓
ASR
  ↓
Shot Alignment
  ↓
Shot Evidence
  ↓
VLM
  ↓
Advertisement Scoring
  ↓
Segment Merging
```

## Long-form

Long videos commonly contain sponsor reads embedded inside normal content.

A sponsor read may not create a major visual scene change. The host may continue speaking in the same visual setting while delivering a commercial message.

Therefore the long-form pipeline uses ASR as an important signal.

```text
ASR
  ↓
Long Scene Detection
  ↓
Long Evidence Extraction
  ↓
Candidate Detection
  ↓
VLM Commercial Verification
  ↓
Segment Merging
```

This is why audio is explicitly included in the long-form pipeline.

---

# 4. Project Structure

```text
ad-segment-detection/
│
├── README.md
├── DESIGN.md
├── EVAL.md
├── ground_truth.json
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── api.py
│   ├── acquire.py
│   ├── pipeline.py
│   ├── probe.py
│   ├── scene.py
│   ├── shots.py
│   ├── shot_ocr.py
│   ├── asr.py
│   ├── shot_asr.py
│   ├── shot_evidence.py
│   ├── vlm_classifier.py
│   ├── ad_scorer.py
│   ├── segment_merger.py
│   │
│   ├── long_scene.py
│   ├── long_evidence.py
│   ├── long_candidate_detector.py
│   ├── long_vlm_classifier.py
│   └── long_segment_merger.py
│
│
└── data/
    ├── videos/
    ├── audio/
    ├── frames/
    └── outputs/
```

The `data/` directory contains generated runtime artifacts and should normally not be committed to Git.

---

# 5. Requirements

Recommended environment:

- Python 3.10+
- FFmpeg
- Git
- Ollama
- Qwen2.5-VL 7B model

The project was developed and tested on Windows.

---

# 6. Installation

## Step 1 — Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ad-segment-detection
```

---

## Step 2 — Create a virtual environment

### Windows

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in the terminal.

---

## Step 3 — Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 7. Install FFmpeg

FFmpeg is required for audio extraction and video processing.

Verify:

```bash
ffmpeg -version
```

If the command is not recognized, install FFmpeg and add its `bin` directory to your system `PATH`.

---

# 8. Install Ollama

The VLM stage uses a local Ollama model.

Install Ollama from:

https://ollama.com/

Verify:

```bash
ollama --version
```

Pull the required model:

```bash
ollama pull qwen2.5vl:7b
```

Verify:

```bash
ollama list
```

You should see:

```text
qwen2.5vl:7b
```

Make sure Ollama is running before using the VLM stages.

---

# 9. Verify the Installation

Run:

```bash
python -c "import fastapi, cv2, whisper; print('Dependencies OK')"
```

Then:

```bash
ffmpeg -version
```

And:

```bash
ollama list
```

If these work, the environment is ready.

---

# 10. Start the API

From the project root:

```bash
python -m src.api
```

Expected output:

```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

# 11. Swagger UI

Open:

```text
http://localhost:8000/docs
```

Swagger UI provides an interactive interface for testing the API.

No separate API client is required for basic testing.

---

# 12. API Endpoints

The API exposes six endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/acquire` | Acquire/download a video |
| POST | `/process` | Acquire and process a video |
| GET | `/result/{video_name}` | Retrieve an existing result |
| GET | `/status/{video_name}` | Check processing status |

---

# 13. GET `/`

Returns basic API information.

Example:

```json
{
  "service": "Ad Segment Detection API",
  "status": "running",
  "version": "1.0.0"
}
```

---

# 14. GET `/health`

Checks whether the API is running.

Example:

```json
{
  "status": "ok"
}
```

---

# 15. POST `/acquire`

The acquire endpoint downloads or acquires the input video and stores it locally.

It does not run advertisement detection.

### Minimal request

```json
{
  "source": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

### Optional filename

```json
{
  "source": "https://www.youtube.com/watch?v=VIDEO_ID",
  "output_name": "my_video"
}
```

The `output_name` field is optional.

### Example response

```json
{
  "success": true,
  "source_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "local_path": "data/videos/VIDEO_ID.mp4",
  "filename": "VIDEO_ID.mp4",
  "error": null
}
```

---

# 16. POST `/process`

This is the primary endpoint.

It performs the complete workflow:

```text
Acquire
  ↓
Probe
  ↓
Automatic Routing
  ↓
Short or Long Pipeline
  ↓
Advertisement Detection
  ↓
Final Result
```

### Minimal request

Only `source` is required:

```json
{
  "source": "https://www.youtube.com/shorts/Ve0zdhTQA4U"
}
```

### Optional filename

```json
{
  "source": "https://www.youtube.com/shorts/Ve0zdhTQA4U",
  "output_name": "my_video"
}
```

Both forms are supported.

---

# 17. Example — YouTube Short

```json
{
  "source": "https://www.youtube.com/shorts/Ve0zdhTQA4U"
}
```

The system automatically:

1. Downloads the video.
2. Probes metadata.
3. Detects the media as short-form.
4. Routes to `short_form`.
5. Runs the short pipeline.
6. Produces the final result.

No manual pipeline-stage execution is required.

---

# 18. Example — Long-form YouTube

```json
{
  "source": "https://www.youtube.com/watch?v=ujFWRFYLGjY"
}
```

The system automatically detects the media type and routes long-form media to the long-form pipeline.

---

# 19. Local File Input

A local file can also be processed:

```json
{
  "source": "data/videos/example.mp4"
}
```

This is useful for locally acquired videos and evaluation data.

---

# 20. Automatic Pipeline Routing

The user does not manually select the pipeline.

The system first probes the media:

```text
Media Metadata
     |
     +---- short ----> short_form
     |
     +---- long_form -> long_form
```

The routing decision is made from the media metadata.

---

# 21. Short-form Pipeline

The short pipeline contains the following stages.

## Step 1 — Scene Detection

The video is analyzed for shot boundaries.

Output:

```text
data/outputs/<video_name>/scenes.json
```

---

## Step 2 — Representative Frames

Representative frames are extracted from each shot.

Current strategy:

```text
early
middle
late
```

Output:

```text
data/outputs/<video_name>/representatives.json
```

and corresponding frame images.

---

## Step 3 — Shot OCR

OCR is applied to representative frames.

It attempts to detect visual commercial signals such as:

- Product names
- Promotional text
- Prices
- Discounts
- Calls to action
- Phone numbers
- URLs
- Brand text

Output:

```text
data/outputs/<video_name>/shot_ocr.json
```

---

## Step 4 — Audio + ASR

The audio track is extracted and transcribed using Whisper.

Outputs:

```text
data/audio/<video_name>.wav
data/outputs/<video_name>/transcript.json
data/outputs/<video_name>/transcript.txt
```

---

## Step 5 — ASR → Shot Alignment

Transcript segments are aligned with the detected shots.

Output:

```text
data/outputs/<video_name>/shot_asr.json
```

---

## Step 6 — Shot Evidence

Visual and audio evidence is combined at shot level.

Output:

```text
data/outputs/<video_name>/shot_evidence.json
```

---

## Step 7 — VLM Classification

Candidate content is verified using Qwen2.5-VL through Ollama.

Possible classifications include:

```text
advertisement
organic_content
uncertain
```

---

## Step 8 — Advertisement Scoring

Signals are combined to calculate advertisement likelihood.

---

## Step 9 — Segment Merging

Adjacent advertisement detections are merged into final segments.

---

## Step 10 — Final Contract

The final result is written as:

```text
data/outputs/<video_name>/result.json
```

---

# 22. Long-form Pipeline

Long-form processing is designed for embedded sponsor reads and other commercial segments that may not have strong visual scene changes.

The pipeline is:

```text
ASR
 ↓
Long Scene Detection
 ↓
Long Evidence Extraction
 ↓
Candidate Detection
 ↓
VLM Verification
 ↓
Segment Merging
 ↓
Final Result
```

---

## Long-form ASR

Whisper transcribes the audio track.

This is important because a long-form sponsor read can be identifiable from speech even when the visual scene remains unchanged.

---

## Long Scene Detection

Long-form video is divided into larger regions.

The purpose is to create manageable regions for downstream evidence analysis.

Output:

```text
data/outputs/<video_name>/long_scenes.json
```

---

## Long Evidence Extraction

A representative frame is extracted from each long-form region.

Current strategy:

```text
middle representative frame
```

Output:

```text
data/outputs/<video_name>/long_evidence.json
data/outputs/<video_name>/long_evidence_frames/
```

---

## Long Candidate Detection

Regions are scored using available evidence and suspicious regions are selected for VLM verification.

Output:

```text
data/outputs/<video_name>/long_candidates.json
```

---

## Long VLM Verification

Candidate regions are passed to Qwen2.5-VL.

The model determines whether a candidate is:

```text
advertisement
organic_content
uncertain
```

It can also provide:

- Confidence
- Advertisement type
- Brand
- Reason
- Supporting signals

Output:

```text
data/outputs/<video_name>/long_vlm_results.json
```

---

## Long Segment Merging

Advertisement candidates are merged into final commercial segments.

Output:

```text
data/outputs/<video_name>/long_segments.json
```

---

# 23. Why Audio Matters

Audio is especially important for long-form content.

For example, consider a host speaking for two minutes in the same studio shot:

```text
Visual:
Same camera angle
Same background
Same person
No major scene change

Audio:
"Today's video is sponsored by..."
"Use code..."
"Visit..."
```

A purely visual scene detector may not identify this as an advertisement.

ASR provides the missing evidence.

Therefore the long-form architecture deliberately uses both:

```text
Visual evidence
+
Audio / ASR evidence
```

The VLM then provides an additional commercial-intent verification stage.

---

# 24. Output

The primary machine-readable output is:

```text
data/outputs/<video_name>/result.json
```

A typical result contains:

```json
{
  "source": {
    "url": "https://...",
    "kind": "short",
    "duration_s": 63.9
  },
  "segments": [
    {
      "start_s": 10.0,
      "end_s": 25.0,
      "ad_type": "midroll_sponsor_read",
      "confidence": 0.90,
      "brand": "Example Brand"
    }
  ],
  "stats": {
    "wall_clock_s": 54.7,
    "frames_sampled": 6,
    "model_calls": 1
  }
}
```

The exact fields depend on the current pipeline implementation.

---

# 25. Retrieve an Existing Result

Use:

```text
GET /result/{video_name}
```

Example:

```text
GET /result/Ve0zdhTQA4U
```

The API reads:

```text
data/outputs/Ve0zdhTQA4U/result.json
```

and returns the stored JSON result.

---

# 26. Check Processing Status

Use:

```text
GET /status/{video_name}
```

Example:

```text
GET /status/Ve0zdhTQA4U
```

Possible states include:

```text
not_found
processing_or_incomplete
completed
```

---

# 27. Running Individual Modules

For debugging, individual stages can also be run directly.

### Probe

```bash
python -m src.probe data/videos/example.mp4
```

### ASR

```bash
python -m src.asr data/videos/example.mp4
```

### Long-form

```bash
python -m src.long_scene data/videos/example.mp4
python -m src.long_evidence data/videos/example.mp4
python -m src.long_candidate_detector data/videos/example.mp4
python -m src.long_vlm_classifier data/videos/example.mp4
python -m src.long_segment_merger data/videos/example.mp4
```

For normal usage, use the API:

```bash
python -m src.api
```

and then:

```text
POST /process
```

The main pipeline handles stage ordering automatically.

---

# 28. Evaluation

The evaluation dataset consists of the five required video inputs.

The exact evaluation procedure, ground-truth labels, metrics, results, and failure analysis are documented in:

```text
EVAL.md
```

Ground-truth annotations are stored in:

```text
ground_truth.json
```

---

# 29. Evaluation Metrics

The evaluation focuses on segment-level detection.

## Precision

```text
Precision = TP / (TP + FP)
```

Measures the fraction of predicted advertisement segments that are correct.

## Recall

```text
Recall = TP / (TP + FN)
```

Measures the fraction of ground-truth advertisements that were detected.

## F1

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

## Segment IoU

For predicted interval `P` and ground-truth interval `G`:

```text
IoU = intersection(P,G) / union(P,G)
```

## Boundary Error

Start and end timestamps are compared with the ground truth to quantify temporal localization error.

The exact matching threshold used for evaluation is documented in `EVAL.md`.

---

# 30. Advertisement Definition

An advertisement is treated as content whose primary purpose is to promote a commercial product, service, brand, paid offering, sponsorship, or explicit commercial transaction.

Potential signals include:

- Promotional language
- Product/service presentation
- Brand mentions
- Calls to action
- Pricing
- Discount codes
- URLs
- Phone numbers
- Promotional graphics
- Sponsor language
- Commercial intent in speech

A single weak signal does not automatically imply that a segment is an advertisement.

---

# 31. Advertisement Types

The system can classify commercial content into categories such as:

```text
preroll
midroll_sponsor_read
product_placement
self_promo
affiliate
platform_inserted
bumper
other
```

Internal classifier categories may differ before final normalization.

---

# 32. VLM Configuration

Current VLM configuration:

```text
Model:
qwen2.5vl:7b

Provider:
Ollama

Temperature:
0

Frame strategy:
middle representative
```

The VLM is run locally through Ollama.

No hosted VLM API key is required for this configuration.

---

# 33. ASR Configuration

Current ASR implementation uses:

```text
Whisper model:
tiny
```

Whisper is used to:

1. Extract speech information from the video.
2. Produce timestamped transcript segments.
3. Align speech with detected shots/regions.
4. Provide evidence for advertisement detection.

---

# 34. Runtime and Performance

Runtime depends on:

- Video duration
- CPU/GPU availability
- Whisper model
- VLM model
- Number of candidate regions
- Number of VLM calls
- Video resolution

The VLM is intentionally used on candidates rather than every possible frame/region to reduce unnecessary model calls.

The pipeline records relevant timing information.

---

# 35. Generated Data

The following directories contain runtime-generated files:

```text
data/videos/
data/audio/
data/frames/
data/outputs/
```

These should normally be excluded from Git.

They can be regenerated by running the pipeline.

---


# 36. Design Documentation

See:

```text
DESIGN.md
```

for the detailed technical design.

It documents:

- Advertisement definition
- Advertisement taxonomy
- Architecture
- Signal selection
- Sampling strategy
- Audio/ASR reasoning
- VLM usage
- Candidate generation
- Segment merging
- Failure modes
- Cost considerations
- Latency considerations
- Scaling considerations
- Future improvements

---

# 37. Evaluation Documentation

See:

```text
EVAL.md
```

for:

- Evaluation dataset
- Ground-truth methodology
- Metrics
- Per-video results
- Segment IoU
- Boundary error
- Failure cases
- Root-cause analysis
- Known limitations

Ground truth is stored in:

```text
ground_truth.json
```

---

# 38. Instagram Reels

Automated Instagram acquisition is not treated as the core engineering problem.

For Reel evaluation data, a locally available video can be supplied directly:

```json
{
  "source": "data/videos/reel1.mp4"
}
```

This allows the detection pipeline to be evaluated independently of platform-specific acquisition restrictions.

---

# 39. Live Video

Live streams differ from VOD because future frames are unavailable.

A live implementation requires windowed processing:

```text
Live Stream
    ↓
Current Window
    ↓
Signal Extraction
    ↓
Candidate Detection
    ↓
Classification
    ↓
Committed Segment
```

A commercial segment crossing a window boundary requires state to be carried between windows.

If a live source is unavailable during evaluation, the exact fallback and its effect on evaluation are documented in `EVAL.md`.

---

# 40. Known Limitations

## Visual Sampling

Representative-frame sampling can miss very short visual advertisements or brief bumper cards.

## OCR

OCR quality depends on:

- Text size
- Font
- Motion
- Image quality
- Language
- Frame quality

## ASR

Whisper accuracy depends on:

- Audio quality
- Language
- Background noise
- Speaker overlap

## VLM

A single representative frame may not contain enough context to determine commercial intent.

This is particularly important for long-form sponsor reads, which is why ASR is used alongside visual evidence.

## Platform Acquisition

Public platform acquisition may change because of:

- Platform restrictions
- Authentication requirements
- Rate limits
- Video availability

## Live Processing

Live processing requires additional state management because segments can cross processing-window boundaries.

---

# 41. Security

Do not commit:

```text
.env
API keys
Passwords
Credentials
Private cookies
Authentication tokens
Downloaded videos
Generated audio
Generated frames
```

The `.gitignore` should exclude generated media and sensitive configuration.

---

# 42. AI Tool Usage

AI coding assistants were used during development for tasks including:

- Code scaffolding
- Debugging
- Refactoring
- Documentation
- Implementation suggestions

The final pipeline structure, integration, signal selection, and evaluation decisions were reviewed and adapted during development.

---

# 43. Quick Start

If you only want to run the system:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ad-segment-detection
```

Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Install:

```cmd
pip install -r requirements.txt
```

Verify:

```cmd
ffmpeg -version
```

Install the VLM:

```cmd
ollama pull qwen2.5vl:7b
```

Start the API:

```cmd
python -m src.api
```

Open:

```text
http://localhost:8000/docs
```

Use:

```text
POST /process
```

with:

```json
{
  "source": "https://www.youtube.com/shorts/Ve0zdhTQA4U"
}
```

The system automatically acquires the video, detects the media type, runs the correct pipeline, and produces:

```text
data/outputs/<video_name>/result.json
```

---

# 44. Submission Checklist

Before submitting, verify that the repository contains:

```text
README.md
DESIGN.md
EVAL.md
ground_truth.json
requirements.txt
.gitignore
src/
viewer/
tests/
```

Also verify that:

```bash
python -m src.api
```

starts successfully and:

```text
http://localhost:8000/docs
```

opens successfully.

Then test:

```text
POST /process
```

using a valid video URL.

---

# 45. Clean-Machine Verification

The most important final test is to reproduce the project from a clean directory.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ad-segment-detection
python -m venv .venv
```

Windows:

```cmd
.venv\Scripts\activate
```

Then:

```cmd
pip install -r requirements.txt
ffmpeg -version
ollama list
python -m src.api
```

Open:

```text
http://localhost:8000/docs
```

Run `/health`.

Then run `/process` with a test video URL.

If the pipeline completes and produces:

```text
data/outputs/<video_name>/result.json
```

the repository is reproducible from the README.

---

