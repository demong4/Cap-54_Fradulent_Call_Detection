# ============================================================
# spark_fusion_stateful_simple_env.py
# Same logic as your original Spark consumer + .env support
# ============================================================

import os
import json
import time
import requests
import numpy as np
import logging

from dotenv import load_dotenv     # <--- added
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (
    StructType, StructField, StringType, ArrayType,
    DoubleType, MapType
)
from rich.console import Console
from rich.text import Text

# ----------------- LOAD ENV -----------------
load_dotenv()

KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "live_calls")
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
FUSION_API = os.getenv("FUSION_API", "http://localhost:5000/api")
TIMEOUT = 40

# ----------------- SPARK INIT -----------------
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 pyspark-shell"
)

spark = (
    SparkSession.builder
    .appName("FraudFusionSimple")
    .config("spark.master", "local[1]")
    .config("spark.driver.memory", "3g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")
logging.getLogger("py4j").setLevel(logging.ERROR)

console = Console()
print("Spark consumer ready.")
print(f"Kafka: {KAFKA_SERVER}")
print(f"Topic: {KAFKA_TOPIC}")
print(f"Fusion API: {FUSION_API}\n")

# ----------------- KAFKA SCHEMA -----------------
schema = StructType([
    StructField("call_id", StringType()),
    StructField("chunk_index", StringType()),
    StructField("event", StringType()),
    StructField("start_timestamp", StringType()),
    StructField("transcript", StringType()),
    StructField("audio_b64", StringType()),
    StructField("mfcc", ArrayType(ArrayType(DoubleType()))),
    StructField("spectrogram", ArrayType(ArrayType(DoubleType()))),
    StructField("timing", MapType(StringType(), StringType()))
])

# ----------------- STATE -----------------
context_buffer = {}   # per-call progressive states

# ----------------- API CALL -----------------
def call_fusion_api(text, mfcc, spec):
    payload = {
        "data": [
            text,
            json.dumps(mfcc.tolist()),
            json.dumps(spec.tolist())
        ]
    }
    try:
        resp = requests.post(FUSION_API, json=payload, timeout=TIMEOUT)
    except Exception as e:
        return None, f"Network error: {e}"

    if resp.status_code != 200:
        return None, f"API Error {resp.status_code}"

    return resp.json()["data"][0], None

# ----------------- PROCESS CHUNK -----------------
def process_chunk(row):
    call_id = row["call_id"]
    idx = int(row["chunk_index"])
    text = (row["transcript"] or "").strip()

    mfcc = np.array(row["mfcc"], dtype=np.float32)
    spec = np.array(row["spectrogram"], dtype=np.float32)

    if call_id not in context_buffer:
        context_buffer[call_id] = {
            "mfccs": [],
            "specs": [],
            "transcripts": []
        }

    buf = context_buffer[call_id]

    buf["mfccs"].append(mfcc)
    buf["specs"].append(spec)
    buf["transcripts"].append(text)

    avg_mfcc = np.mean(buf["mfccs"], axis=0)
    avg_spec = np.mean(buf["specs"], axis=0)
    full_text = " ".join(buf["transcripts"])[:512]

    res, err = call_fusion_api(full_text, avg_mfcc, avg_spec)
    if err:
        console.print(f"[API Error] {err}")
        return

    score = float(res.get("final_prob", 0.0))
    bert_p = res.get("bert_prob", 0.0)
    mfcc_p = res.get("mfcc_prob", 0.0)
    spec_p = res.get("spec_prob", 0.0)

    status = "FRAUDULENT" if score > 0.5 else "Normal"
    style = "bold red" if score > 0.5 else "green"

    line = Text(f"[{call_id}:{idx}] ", style="cyan")
    line.append(f"Transcript: {full_text[:80]}{'...' if len(full_text)>80 else ''}\n")
    line.append(status, style=style)
    line.append(f" | Prob={score:.3f} ({bert_p:.3f},{mfcc_p:.3f},{spec_p:.3f})")

    console.print(line)

# ----------------- PROCESS CALL_END -----------------
def process_call_end(row):
    call_id = row["call_id"]

    if call_id not in context_buffer:
        console.print(f"[Warning] CALL_END with no state for {call_id}")
        return

    buf = context_buffer[call_id]

    full_text = " ".join(buf["transcripts"])[:2048]
    avg_mfcc = np.mean(buf["mfccs"], axis=0)
    avg_spec = np.mean(buf["specs"], axis=0)

    res, err = call_fusion_api(full_text, avg_mfcc, avg_spec)
    if err:
        console.print(f"[Final API Error] {err}")
        del context_buffer[call_id]
        return

    score = float(res.get("final_prob", 0.0))
    bert_p = res.get("bert_prob", 0.0)
    mfcc_p = res.get("mfcc_prob", 0.0)
    spec_p = res.get("spec_prob", 0.0)

    status = "FRAUDULENT" if score > 0.5 else "Normal"
    style = "bold red" if score > 0.5 else "green"

    console.print("\n================ FINAL CALL SUMMARY ================")
    console.print(f"Call ID: {call_id}")
    console.print(f"Total Segments: {len(buf['transcripts'])}\n")

    console.print("Final Transcript:")
    console.print(full_text + "\n")

    final_line = Text(f"Final Decision: {status}", style=style)
    console.print(final_line)
    console.print(f"Probability={score:.3f}   (bert={bert_p:.3f}, mfcc={mfcc_p:.3f}, spec={spec_p:.3f})")
    console.print("====================================================\n")

    del context_buffer[call_id]

# ----------------- DISPATCHER -----------------
def handle_row(row):
    if row.get("event") == "CALL_END":
        process_call_end(row)
    else:
        process_chunk(row)

# ----------------- BATCH HANDLER -----------------
def process_batch(df, epoch_id):
    for row in df.collect():
        handle_row(row.asDict())

# ----------------- START STREAM -----------------
df_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SERVER)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

parsed = (
    df_stream.select(from_json(col("value").cast("string"), schema).alias("data"))
    .select("data.*")
)

query = (
    parsed.writeStream
    .foreachBatch(process_batch)
    .outputMode("append")
    .start()
)

query.awaitTermination()
