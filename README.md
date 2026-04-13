# AI-Enhanced Video Streaming and Recommendation System

This repository implements the **control plane** for the architecture described in the project brief: dynamic adaptive quality from live network and user context, observability, forecasting for capacity signals, and deployment patterns for caching, load balancing, multi-bitrate delivery, and edge/P2P extensions.

## Architecture mapping

| Concept | Implementation |
|--------|----------------|
| Real-time network monitoring | `POST /network/probe` stores bandwidth, latency, congestion; **Prometheus** scrapes `GET /metrics`; **Grafana** dashboards via Docker |
| SNMP | Point **snmp_exporter** + Prometheus at your gear; forward summaries into `/network/probe` with `source=snmp` (same schema) |
| Viewer preferences | **PostgreSQL** via `PUT/GET /users/{id}/preferences` (preferred max resolution, buffering tolerance) |
| AI decision-making | **Tabular Q-learning** agent in `backend/app/services/quality_rl.py` (swap for TensorFlow/PyTorch/SageMaker models behind the same API) |
| Quality adjustment | `POST /quality/recommend` returns ladder rung; **FFmpeg** script produces HLS renditions; players use **DASH/HLS** ABR |
| Bottleneck prediction | `POST /network/forecast` — linear trend baseline; replace with **LSTM/ARIMA** + Spark/feature store as needed |
| Caching / LB | Example **NGINX** and **HAProxy** configs in `configs/` |
| P2P / latency | **WebRTC** / **libp2p** are not bundled here; run edge caches (e.g. Greengrass) in front of origin and optionally add a P2P assist layer per your CDN strategy |
| SD-WAN | Operational routing layer; metrics still land in probes/Prometheus |
| Video content analysis | `POST /content/analyze` — **CNN** (ResNet-18 / ImageNet) on sampled frames, **GRU** over frame embeddings for transition / scene-change cues, **NLP** (TF–IDF + theme seeds; transformer-ready) on optional transcript |

## Quick start (Docker)

```bash
docker compose up --build
```

- API: http://localhost:8000/docs  
- Prometheus: http://localhost:9090  
- Grafana: http://localhost:3000 (admin / admin)  

### Example flow

1. Seed preferences: `PUT /users/demo/preferences` with `preferred_max_height` and `buffering_tolerance_sec`.
2. Emit probes (from your network agents): `POST /network/probe` with `bandwidth_mbps`, `latency_ms`, `congestion`.
3. Ask for a rung: `POST /quality/recommend` with `user_id`, current network stats; optional `use_forecast` uses stored history.
4. After playback, send `POST /quality/feedback` with stalls and played resolution to update the RL table.

### Video content analysis

`POST /content/analyze` (multipart): field `video` = file, optional `transcript` = form text. Requires **ffmpeg** in the container (enabled in `backend/Dockerfile`) and **PyTorch / torchvision / scikit-learn** (`requirements-analysis.txt`). TensorFlow can mirror the same split: `tf.keras.applications` for CNN, `tf.keras.layers.RNN` for sequence modeling, and your NLP stack for dialogue.

## FFmpeg adaptive renditions

```bash
chmod +x scripts/transcode_abr.sh
./scripts/transcode_abr.sh /path/to/source.mp4 ./media/out
```

Map manifest rung heights to the ladder in `quality_rl.LADDER` so server and client stay aligned.

## Production notes

- Persist RL policy (Redis/DB) if you run multiple API replicas.
- Train deep RL / supervised rankers offline; serve checkpoints via the same `recommend` handler.
- For **MongoDB** instead of Postgres, swap SQLAlchemy models for an ODM and keep the same REST shapes.

## License

See repository root for project documentation (`AI-Enhanced Video Streaming Service.pdf` / `.docx`).
