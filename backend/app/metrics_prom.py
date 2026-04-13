from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

QUALITY_DECISIONS = Counter(
    "streaming_quality_decisions_total",
    "ABR quality decisions",
    ["recommended_height"],
)
NETWORK_PROBE_INGEST = Counter("streaming_network_probes_ingested_total", "Network probe samples stored")
FORECAST_REQUESTS = Counter("streaming_forecast_requests_total", "Forecast API calls")
LAST_BANDWIDTH_MBPS = Gauge("streaming_last_bandwidth_mbps", "Most recent probe bandwidth Mbps")
LAST_LATENCY_MS = Gauge("streaming_last_latency_ms", "Most recent probe latency ms")
DECISION_LATENCY = Histogram(
    "streaming_quality_decision_seconds",
    "Time to compute quality recommendation",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
CONTENT_ANALYSIS_REQUESTS = Counter(
    "streaming_content_analysis_requests_total",
    "Video content analysis API calls",
    ["status"],
)
CONTENT_ANALYSIS_SECONDS = Histogram(
    "streaming_content_analysis_seconds",
    "End-to-end content analysis duration",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)
RECOMMENDATION_REQUESTS = Counter(
    "streaming_recommendation_requests_total",
    "Hybrid recommendation API calls",
    ["fallback"],
)
RECOMMENDATION_FIT_SECONDS = Histogram(
    "streaming_recommendation_fit_seconds",
    "Time to refit MF + neural scorer",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
INGESTION_EVENTS = Counter(
    "streaming_ingestion_events_total",
    "Unified data aggregator ingest calls",
    ["source"],
)
USER_INTERACTIONS = Counter(
    "streaming_user_interactions_total",
    "User interaction feedback events",
    ["interaction_type"],
)


def metrics_response():
    return generate_latest(), CONTENT_TYPE_LATEST
