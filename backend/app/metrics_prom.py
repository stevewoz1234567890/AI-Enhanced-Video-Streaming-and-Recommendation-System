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


def metrics_response():
    return generate_latest(), CONTENT_TYPE_LATEST
