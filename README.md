# Real-Time Fraud Call Detection

A real-time, multimodal fraud-detection system that listens to a live phone call and decides — while the call is still going on — whether the caller is running a scam. Audio is captured from the microphone, split into speech segments, transcribed, converted into acoustic features, streamed through Kafka, and scored by a fusion model that combines what was *said* (BERT) with how it *sounded* (MFCC CNN + spectrogram CNN ensemble).

The system produces a running verdict for every speech segment and a final decision when the call ends.

First-author research work — *"Real-Time Fraud Call Detection via Multimodal Deep Learning Fusion"*, accepted and presented at **CSOC 2026**.

**97% accuracy · 99.3% AUC**

---

## How It Works

```
   Microphone
       |
       v
  [ WebRTC VAD ]  ---> speech segments only (silence discarded)
       |
       v
  Feature extraction + transcription
   - MFCC + delta + delta2 + spectral contrast
   - 224x224 mel spectrogram
   - Speech-to-text transcript
       |
       v
  Kafka topic: live_calls          (key = call_id, guarantees per-call ordering)
       |
       v
  Spark Structured Streaming consumer
   - keeps per-call state
   - averages features across all segments so far
   - concatenates the transcript so far
       |
       v
  Fusion Inference API
   - BERT        -> text fraud probability
   - MFCC CNN    -> acoustic fraud probability
   - Spec CNNs   -> spectrogram fraud probability
   - weighted fusion (0.6 / 0.2 / 0.2)
       |
       v
  Live verdict per segment  +  final call summary
```

---

## Features

### Real-Time Audio Pipeline
- Continuous microphone capture at 16 kHz in 30 ms frames
- Voice Activity Detection (WebRTC VAD) isolates speech and discards silence, so only meaningful audio is processed
- Ring-buffered trigger logic that opens a segment on sustained speech and closes it on sustained silence
- Multi-threaded design — capture, segmentation, and feature extraction run concurrently so the microphone is never blocked

### Multimodal Feature Extraction
- **Text:** live speech-to-text transcription of every segment
- **MFCC:** 40 MFCC coefficients plus first and second order deltas and spectral contrast, length-normalised to a fixed 200-frame window
- **Spectrogram:** 128/224-band mel spectrogram converted to dB and min-max normalised to a 224x224 image
- Raw audio carried alongside as base64 WAV for replay and audit

### Streaming Backbone
- Kafka as the transport between capture and inference, with `call_id` as the partition key so every segment of a call lands on the same partition in order
- Large-payload producer configuration for feature-rich messages
- Per-segment latency instrumentation (feature time, transcription time, total time) embedded in every message
- Explicit `CALL_END` event that triggers the final verdict

### Stateful Stream Processing
- Spark Structured Streaming consumer with `foreachBatch` processing
- Per-call context buffer that accumulates transcripts and features across segments
- **Progressive scoring** — each new segment is scored against everything heard so far, not in isolation, so confidence sharpens as the call develops
- Final call summary on `CALL_END`: full transcript, segment count, per-modality probabilities and the fused decision
- Colour-coded live console output (green = normal, red = fraudulent)

### Fusion Model
- **BERT** fine-tuned on call transcripts for semantic fraud cues ("KYC expired", "verify now", urgency and authority patterns)
- **MFCC CNN** with a custom temporal-attention layer over the acoustic sequence
- **Spectrogram CNN ensemble** — DenseNet169, EfficientNet-B3 and EfficientNet-B4 averaged
- **Fusion layer** selected by 5-fold stratified cross-validation across a weight grid and competing meta-models (weighted average, logistic regression, MLP); the weighted average at 0.6 / 0.2 / 0.2 won
- Served behind an HTTP inference endpoint so the streaming layer stays decoupled from the model runtime

---

## Tech Stack

### Streaming & Infrastructure
- Apache Kafka
- Apache Spark (Structured Streaming, PySpark)
- REST inference service

### Machine Learning
- PyTorch, Transformers (BERT)
- TensorFlow / Keras (MFCC CNN + temporal attention)
- timm (DenseNet169, EfficientNet-B3, EfficientNet-B4)
- scikit-learn (fusion meta-models, cross-validation, metrics)

### Audio Processing
- librosa, soundfile
- sounddevice (live capture)
- webrtcvad (voice activity detection)
- SpeechRecognition (transcription)

### Utilities
- NumPy, pandas
- rich (live console rendering)
- python-dotenv

---

## Project Structure

```
Cap-54_Fradulent_Call_Detection
├── KafkaProducer.py       # Live mic capture, VAD segmentation,
│                          # feature extraction, transcription,
│                          # Kafka publishing with per-call keys
│
├── SparkConsumer.py       # Spark Structured Streaming consumer,
│                          # per-call stateful buffering,
│                          # fusion API calls, live + final verdicts
│
├── Models_Kaggle.ipynb    # Dataset cleaning and stratified splits,
│                          # BERT fine-tuning, MFCC CNN, spectrogram
│                          # CNN ensemble, fusion meta-model search,
│                          # inference service definition
│
└── README.md
```

---

## Getting Started

### Clone the repository

```bash
git clone https://github.com/demong4/Cap-54_Fradulent_Call_Detection.git
cd Cap-54_Fradulent_Call_Detection
```

### Install dependencies

```bash
pip install kafka-python pyspark numpy librosa soundfile sounddevice \
            webrtcvad SpeechRecognition requests rich python-dotenv
```

> On Linux, `sounddevice` needs PortAudio: `sudo apt install libportaudio2`

### Start Kafka

```bash
# start ZooKeeper / KRaft and the broker, then create the topic
kafka-topics.sh --create --topic live_calls \
                --bootstrap-server localhost:9092 \
                --partitions 3 --replication-factor 1
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
KAFKA_SERVER=localhost:9092
KAFKA_TOPIC=live_calls
FUSION_API=http://localhost:5000/api
```

---

## Run the Application

Start the consumer first, so no segments are missed:

```bash
python SparkConsumer.py
```

Then start the live capture:

```bash
python KafkaProducer.py
```

Speak into the microphone. Each detected speech segment prints a live verdict. Press `Ctrl+C` to end the call and emit the final summary.

---

## Message Schema

Every segment published to `live_calls`:

| Field | Type | Description |
|---|---|---|
| `call_id` | string | Stable id for the call, also the Kafka partition key |
| `chunk_index` | int | Sequence number of the segment within the call |
| `event` | string | `CHUNK` or `CALL_END` |
| `start_timestamp` | string | UTC timestamp of the segment |
| `transcript` | string | Speech-to-text output for the segment |
| `mfcc` | float[][] | MFCC + deltas + spectral contrast, fixed 200-frame window |
| `spectrogram` | float[224][224] | Normalised mel spectrogram |
| `audio_b64` | string | Base64-encoded WAV of the raw segment |
| `timing` | object | `feature_ms`, `transcribe_ms`, `total_ms` |

---

## Inference API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/fusion_predict` | Score a transcript + MFCC + spectrogram triple |

Request:

```json
{ "data": ["KYC expired please verify now", "<mfcc json>", "<spectrogram json>"] }
```

Response — per-modality probabilities plus the fused score:

```json
{
  "bert_prob": 0.981,
  "mfcc_prob": 0.774,
  "spec_prob": 0.812,
  "final_prob": 0.906
}
```

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 97% |
| AUC | 99.3% |
| Fusion weights (BERT / MFCC / Spectrogram) | 0.60 / 0.20 / 0.20 |
| Validation | 5-fold stratified cross-validation |
| Dataset split | 75 / 15 / 15 stratified train-validation-test |

---

## Highlights

- End-to-end real-time pipeline: microphone to verdict, no offline batch step
- Multimodal fusion across text, cepstral and spectral representations
- Voice Activity Detection so compute is spent only on speech
- Per-call ordering guaranteed through Kafka partition keys
- Stateful streaming with progressive, sharpening confidence
- Model selection driven by cross-validated comparison, not a single lucky split
- Decoupled inference service — the model can be retrained or replaced without touching the streaming layer
- Published and presented at CSOC 2026 as first author

---

## Future Improvements

- Speaker diarisation to separate caller and receiver
- On-device transcription to remove the external speech-to-text dependency
- Whisper-based transcription for accent and noise robustness
- Model registry and versioned inference endpoints
- Streaming aggregation windows in place of full-history averaging
- Dashboard for live call monitoring and historical analytics
- Multilingual support for Indian regional languages
- Automatic call blocking or warning hooks on a high-confidence verdict

---

## Citation

```
G. R. Halembre et al., "Real-Time Fraud Call Detection via Multimodal
Deep Learning Fusion," CSOC 2026.
```

---

## License

This project is licensed under the MIT License.

---

## Author

**Ganesh R Halembre**

- GitHub: https://github.com/demong4
- LinkedIn: https://linkedin.com/in/ganesh-r-halembre
