                     ┌──────────────┐
                     │  Video URL   │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ Acquisition  │
                     └──────┬───────┘
                            │
                            ▼
                       ┌─────────┐
                       │  Probe  │
                       └────┬────┘
                            │
                  ┌─────────┴─────────┐
                  │                   │
              SHORT                 LONG
                  │                   │
                  ▼                   ▼
              Scenes               ASR
                  │                   │
              Frames             Long Scenes
                  │                   │
                OCR             Long Evidence
                  │                   │
                ASR             Candidates
                  │                   │
             Shot Evidence         VLM
                  │                   │
                 VLM             Segment Merge
                  │                   │
             Ad Scoring               │
                  │                   │
             Segmentation             │
                  └─────────┬─────────┘
                            ▼
                     Final Contract
                            │
                            ▼
                       result.json