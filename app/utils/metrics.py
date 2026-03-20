from prometheus_client import Histogram

MODEL_INFERENCE_LATENCY_SECONDS = Histogram(
    "model_inference_latency_seconds",
    "Latency of model inference in seconds",
    ["model_id", "model_type"],
    buckets=(
        0.001, 0.005, 0.01, 0.025, 0.05,
        0.1, 0.25, 0.5, 1.0, 2.5, 5.0
    ),
)