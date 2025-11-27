# =====================================================================
# fp_producer_vad_cleanlog.py – SAME AS ORIGINAL + PARTITION KEY
# =====================================================================

import io
import time
import uuid
import json
import queue
import base64
import threading
import collections

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import speech_recognition as sr
import webrtcvad

from kafka import KafkaProducer

# ----------------------- CONFIG --------------------------------------
KAFKA_SERVER = "172.26.223.173:9092"
KAFKA_TOPIC = "live_calls"

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

VAD = webrtcvad.Vad(2)

CALL_ID = f"call_{uuid.uuid4().hex[:8]}"

print(f"Mic ready — streaming as {CALL_ID}")

# ----------------------- KAFKA SETUP ---------------------------------
producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    key_serializer=lambda k: k.encode(),           # <-- ADDED
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    max_request_size=52428800
)

# ----------------------- THREAD QUEUES --------------------------------
frame_queue = queue.Queue()
speech_segments = queue.Queue()

recognizer = sr.Recognizer()

# ----------------------- AUDIO CALLBACK -------------------------------
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[AUDIO STATUS] {status}")
    audio_bytes = (indata[:, 0] * 32767).astype(np.int16).tobytes()
    frame_queue.put(audio_bytes)

# ----------------------- VAD COLLECTOR --------------------------------
def vad_collector():
    ring_buffer = collections.deque(maxlen=10)
    triggered = False
    voiced_frames = []

    while True:
        frame = frame_queue.get()
        is_speech = VAD.is_speech(frame, SAMPLE_RATE)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])

            if num_voiced > 0.6 * ring_buffer.maxlen:
                triggered = True
                for f, _ in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()

        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])

            if num_unvoiced > 0.6 * ring_buffer.maxlen:
                triggered = False
                segment = b"".join(voiced_frames)
                speech_segments.put(segment)
                voiced_frames = []

# ----------------------- FEATURE EXTRACTION ----------------------------
def extract_mfcc(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc   = librosa.feature.mfcc(S=mel_db, sr=sr, n_mfcc=40)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    sc     = librosa.feature.spectral_contrast(S=mel, sr=sr)
    feat   = np.concatenate([mfcc, delta, delta2, sc[:13]], axis=0)
    feat   = librosa.util.fix_length(feat, size=200, axis=1)
    return feat.T.astype(np.float32).tolist()

def extract_spec(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = librosa.util.fix_length(mel_db, size=224, axis=1)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    return mel_norm.tolist()

# ----------------------- SEGMENT WORKER ----------------------------
def segment_worker():
    chunk_idx = 0

    while True:
        audio_bytes = speech_segments.get()
        t_start = time.time()

        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0

        # ----- Feature Extraction -----
        t_feat0 = time.time()
        mfcc_feat = extract_mfcc(audio_np, SAMPLE_RATE)
        spec_feat = extract_spec(audio_np, SAMPLE_RATE)
        t_feat1 = time.time()

        # ----- Transcription -----
        t_tr0 = time.time()
        audio_data = sr.AudioData(audio_bytes, SAMPLE_RATE, 2)
        try:
            text = recognizer.recognize_google(audio_data)
        except:
            text = ""
        t_tr1 = time.time()

        if text.strip() == "":
            continue

        # ----- Encode audio -----
        buf = io.BytesIO()
        sf.write(buf, audio_np, SAMPLE_RATE, format="WAV")
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode("utf-8")

        total_ms = (time.time() - t_start) * 1000
        feature_ms = (t_feat1 - t_feat0) * 1000
        transcribe_ms = (t_tr1 - t_tr0) * 1000

        payload = {
            "call_id": CALL_ID,
            "chunk_index": chunk_idx,
            "event": "CHUNK",
            "start_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
            "transcript": text,
            "mfcc": mfcc_feat,
            "spectrogram": spec_feat,
            "audio_b64": audio_b64,
            "timing": {
                "feature_ms": round(feature_ms, 2),
                "transcribe_ms": round(transcribe_ms, 2),
                "total_ms": round(total_ms, 2)
            }
        }

        # -----------------------
        # SEND WITH PARTITION KEY
        # -----------------------
        producer.send(
            KAFKA_TOPIC,
            key=CALL_ID,                   # <-- ADDED (forces ordering)
            value=payload
        )
        producer.flush()

        # -----------------------
        # CLEAN 2-LINE LOG OUTPUT
        # -----------------------
        print(f"Chunk {chunk_idx} : Transcript: {text}")
        print(
            f"feature_ms={payload['timing']['feature_ms']} | "
            f"transcribe_ms={payload['timing']['transcribe_ms']} | "
            f"total_ms={payload['timing']['total_ms']}"
        )
        print()

        chunk_idx += 1

# ----------------------- START EVERYTHING ------------------------------
stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=FRAME_SAMPLES,
    callback=audio_callback
)

threading.Thread(target=vad_collector, daemon=True).start()
threading.Thread(target=segment_worker, daemon=True).start()

print("Listening using VAD... Ctrl+C to stop.\n")

try:
    with stream:
        while True:
            time.sleep(0.1)

except KeyboardInterrupt:
    # -----------------------
    # SEND CALL_END WITH KEY
    # -----------------------
    producer.send(
        KAFKA_TOPIC,
        key=CALL_ID,                      # <-- ADDED
        value={
            "call_id": CALL_ID,
            "event": "CALL_END",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
    )
    producer.flush()
    print(f"Sent CALL_END for {CALL_ID}")
