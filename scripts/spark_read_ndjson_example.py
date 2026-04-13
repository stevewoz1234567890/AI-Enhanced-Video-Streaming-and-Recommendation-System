#!/usr/bin/env python3
"""
Example: load training export NDJSON in Apache Spark for distributed feature prep.

  curl -s "http://localhost:8000/analytics/export/training?limit_per_stream=10000" \\
    -o /tmp/streaming_training.ndjson

  spark-submit --packages org.apache.spark:spark-sql_2.12:3.5.0 \\
    scripts/spark_read_ndjson_example.py /tmp/streaming_training.ndjson Without Spark, the script falls back to a simple Python line counter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def read_with_spark(path: str) -> None:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("streaming-training-export").getOrCreate()
    df = spark.read.json(path)
    df.printSchema()
    df.groupBy("stream").count().show()


def read_simple(path: str) -> None:
    counts: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            s = row.get("stream", "unknown")
            counts[s] = counts.get(s, 0) + 1
    print("counts_by_stream:", counts)


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "/tmp/streaming_training.ndjson"
    if not Path(p).is_file():
        print("File not found:", p, file=sys.stderr)
        sys.exit(1)
    try:
        read_with_spark(p)
    except ImportError:
        read_simple(p)
