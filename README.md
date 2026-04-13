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
| Viewing habits & recommendations | `POST /viewing/event` logs watch time / ratings; `GET /recommendations/{user_id}` blends **matrix factorization** (TruncatedSVD / collaborative filtering) with a **PyTorch MLP**; cold-start uses popularity. TensorFlow can replace the MLP on the same latent features |
| Streaming framework integration | **Ingestion:** `POST /ingest/event` and `/ingest/batch` (video servers, devices, edges, infra). **ABR + edge:** `POST /edges/register`, `POST /delivery/optimize` (predictive routing hints, transcoding/edge alignment). **Feedback loop:** `POST /feedback/interaction` (RL + recommender cache invalidation). |

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

### Viewing habits & recommendations

- `POST /viewing/event` — JSON body: `user_id`, `content_id`, `watch_seconds`, optional `rating` (1–5).
- `GET /recommendations/{user_id}?top_k=10&exclude_watched=true` — hybrid scores from **scikit-learn** TruncatedSVD + **PyTorch** MLP; needs the same ML stack as content analysis (Docker image already installs it).

### Integration with streaming stacks

- **Data aggregator:** `POST /ingest/event` with `source` = `video_server` | `user_device` | `edge` | `infra`, plus `event_type` and JSON `payload`. Use `event_type: network_probe` and payload keys `bandwidth_mbps`, `latency_ms`, `congestion` to mirror `POST /network/probe` into the same forecast/ABR history. High-volume agents can use `POST /ingest/batch` (up to 500 events).
- **Edge + ABR:** Register packaging/CDN edges with `POST /edges/register` (`edge_id`, `region`, `base_url`, `capacity_units`). Send `source: infra`, `event_type: edge_load`, `origin_id: <edge_id>`, payload `current_load` / `healthy` to simulate LB/cache telemetry. `POST /delivery/optimize` returns ABR ladder, chosen edge, forecast risk, and hints for cache, LB, P2P, and routing (wire to NGINX/HAProxy/Varnish as needed).
- **Feedback loop:** `POST /feedback/interaction` stores structured interactions; `playback_quality_feedback` updates the tabular RL agent (same fields as `POST /quality/feedback` inside `payload`); `explicit_dislike`, `not_interested`, `rate`, and `rating` invalidate the hybrid recommender cache so the next `GET /recommendations` refits.

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
