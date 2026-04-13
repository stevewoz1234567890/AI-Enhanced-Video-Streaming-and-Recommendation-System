"""GDPR/CCPA-style helpers: consent checks, portable export, erasure."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    IngestionEvent,
    UserInteraction,
    UserPreference,
    UserPrivacySettings,
    ViewingEvent,
)


async def personalization_allowed(session: AsyncSession, user_id: str) -> bool:
    row = await session.get(UserPrivacySettings, user_id)
    if row is None:
        return True
    return bool(row.consent_personalization)


async def require_personalization(session: AsyncSession, user_id: str) -> None:
    if not await personalization_allowed(session, user_id):
        raise HTTPException(
            status_code=403,
            detail="Personalization is disabled for this account. Enable consent_personalization via PUT /privacy/consent.",
        )


async def export_portable_data(session: AsyncSession, user_id: str) -> dict:
    prefs = await session.get(UserPreference, user_id)
    priv = await session.get(UserPrivacySettings, user_id)
    vres = await session.execute(select(ViewingEvent).where(ViewingEvent.user_id == user_id))
    viewing = [
        {
            "id": v.id,
            "content_id": v.content_id,
            "watch_seconds": v.watch_seconds,
            "rating": v.rating,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        }
        for v in vres.scalars().all()
    ]
    ires = await session.execute(select(UserInteraction).where(UserInteraction.user_id == user_id))
    interactions = [
        {
            "id": u.id,
            "interaction_type": u.interaction_type,
            "content_id": u.content_id,
            "payload": u.payload,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }
        for u in ires.scalars().all()
    ]
    return {
        "user_id": user_id,
        "preferences": (
            {
                "preferred_max_height": prefs.preferred_max_height,
                "buffering_tolerance_sec": prefs.buffering_tolerance_sec,
                "updated_at": prefs.updated_at.isoformat() if prefs.updated_at else None,
            }
            if prefs
            else None
        ),
        "privacy_settings": (
            {
                "consent_personalization": priv.consent_personalization,
                "consent_analytics": priv.consent_analytics,
                "consent_model_training": priv.consent_model_training,
                "policy_version_ack": priv.policy_version_ack,
                "updated_at": priv.updated_at.isoformat() if priv.updated_at else None,
            }
            if priv
            else None
        ),
        "viewing_events": viewing,
        "user_interactions": interactions,
    }


async def erase_user_data(session: AsyncSession, user_id: str) -> None:
    await session.execute(delete(ViewingEvent).where(ViewingEvent.user_id == user_id))
    await session.execute(delete(UserInteraction).where(UserInteraction.user_id == user_id))
    await session.execute(delete(UserPreference).where(UserPreference.user_id == user_id))
    await session.execute(delete(UserPrivacySettings).where(UserPrivacySettings.user_id == user_id))
    await session.execute(
        delete(IngestionEvent).where(
            IngestionEvent.origin_id == user_id,
            IngestionEvent.source == "user_device",
        )
    )
