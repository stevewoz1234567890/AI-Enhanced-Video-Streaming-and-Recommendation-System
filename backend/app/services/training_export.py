"""NDJSON-style export for Spark / EMR / Glue batch jobs and cloud data lakes."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import IngestionEvent, NetworkSample, UserInteraction, ViewingEvent


async def collect_training_records(
    session: AsyncSession,
    streams: list[str],
    limit_per_stream: int,
) -> list[dict]:
    out: list[dict] = []
    stream_set = {s.strip().lower() for s in streams if s.strip()}

    if "network" in stream_set or "network_samples" in stream_set:
        res = await session.execute(
            select(NetworkSample).order_by(NetworkSample.id.desc()).limit(limit_per_stream)
        )
        for s in res.scalars().all():
            out.append(
                {
                    "stream": "network_sample",
                    "id": s.id,
                    "ts": s.ts.isoformat() if s.ts else None,
                    "bandwidth_mbps": s.bandwidth_mbps,
                    "latency_ms": s.latency_ms,
                    "congestion": s.congestion,
                    "source": s.source,
                }
            )

    if "viewing" in stream_set or "viewing_events" in stream_set:
        res = await session.execute(
            select(ViewingEvent).order_by(ViewingEvent.id.desc()).limit(limit_per_stream)
        )
        for v in res.scalars().all():
            out.append(
                {
                    "stream": "viewing_event",
                    "id": v.id,
                    "user_id": v.user_id,
                    "content_id": v.content_id,
                    "watch_seconds": v.watch_seconds,
                    "rating": v.rating,
                    "created_at": v.created_at.isoformat() if v.created_at else None,
                }
            )

    if "ingestion" in stream_set or "ingestion_events" in stream_set:
        res = await session.execute(
            select(IngestionEvent).order_by(IngestionEvent.id.desc()).limit(limit_per_stream)
        )
        for e in res.scalars().all():
            out.append(
                {
                    "stream": "ingestion_event",
                    "id": e.id,
                    "source": e.source,
                    "origin_id": e.origin_id,
                    "event_type": e.event_type,
                    "payload": e.payload,
                    "client_ts": e.client_ts.isoformat() if e.client_ts else None,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
            )

    if "interactions" in stream_set or "user_interactions" in stream_set:
        res = await session.execute(
            select(UserInteraction).order_by(UserInteraction.id.desc()).limit(limit_per_stream)
        )
        for u in res.scalars().all():
            out.append(
                {
                    "stream": "user_interaction",
                    "id": u.id,
                    "user_id": u.user_id,
                    "interaction_type": u.interaction_type,
                    "content_id": u.content_id,
                    "payload": u.payload,
                    "created_at": u.created_at.isoformat() if u.created_at else None,
                }
            )

    return out
