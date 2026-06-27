---
title: Aletheia's Quest Leaderboard
emoji: 🔎
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Aletheia's Quest — Leaderboard Space

This is the HuggingFace **Docker Space** front-matter + deploy notes for the
runner in this repo. The Space builds the root `Dockerfile`, which installs the
`aletheia_runner` package and launches the FastAPI app on port 7860.

## Deploy

1. Create a **Docker** Space in the NDIF org.
2. Push this repo to the Space (its root `README.md` must carry the front-matter
   above — copy this file to the repo root when deploying, or develop the Space
   as a mirror of this repo).
3. Set Space **secrets**:
   - `HF_TOKEN` — org token with read access to the gated eval + labels datasets.
4. Point `runner.yaml` `datasets:` at the real private datasets.
5. (Recommended) Attach **persistent storage** and keep `RESULTS_URI=/data/results.jsonl`
   so the leaderboard survives restarts.

## Endpoints

- `GET /` — leaderboard page
- `GET /api/leaderboard` — JSON
- `GET /api/health` — configured datasets
- `POST /submit` — multipart `team`, `file` (zip), optional `token`

## Not yet wired

- Pushing results to a HF results dataset (the "bucket") — currently JSONL on disk.
- Egress allowlist enforcement (`runner.yaml: egress_allowlist`) — parsed, not enforced.
- Submit auth / rate limiting and eval-vs-test phase gating.
