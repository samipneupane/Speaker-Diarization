# Speaker Diarization in Nepali Language

This project implements end-to-end Speaker Diarization Nepali (and related) languages, using the following two models:

- **EEND‑EDA**: Encoder–decoder attractor networks with self‑attention for handling unknown speaker counts.
- **DiaPer**: Perceiver‑based attractor networks offering  better performance, more accurate speaker count estimation, and faster inference time.

The system works well with 2 to 4 speakers, even in mixed situations, reaching less than 5% DER on Nepali test sets and performing nearly as well as the best systems on English datasets.

## Reference Implementations
In our work, we used the model architecture and modify the necessary configurations from the official GitHub repositories: [BUTSpeechFIT/EEND](https://github.com/BUTSpeechFIT/EEND) and [BUTSpeechFIT/DiaPer](https://github.com/BUTSpeechFIT/DiaPer). These references provided a solid foundation for implementing and evaluating our speaker diarization models.

## Dataset Sources

| Dataset                                                                              | Language | #Speakers |
| ------------------------------------------------------------------------------------ | -------- | --------- |
| [LibriSpeech](https://www.openslr.org/12/)                                           | English  | 921       |
| [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)                           | English  | 1,211     |
| [Nepali Female Speakers](https://www.openslr.org/43/)                                | Nepali   | 18        |
| [Hindi Audio (Shukla 2020)](https://github.com/ShivamShukla123/Hindi-Speech-Dataset/)| Hindi    | 100       |

The multilingual mix ensures robust generalization across acoustic conditions.

## Dataset Preparation Methodology

1. **Voice Activity Detection (VAD)**: Employed `webrtcvad` to remove non‑speech frames, reducing false alarms
2. **Synthetic Mixtures**:
   - **2‑, 3‑, 4‑speaker**: Randomly selected single‑speaker clips, concatenated with 0.5–1s silences to mimic turn‑taking
   - **Mixed scenarios**: Combined segments containing varying speaker counts
3. **Kaldi Metadata**: Generated `wav.scp`, `utt2spk`, `reco2dur`, `segments`, and `merge` files for seamless data loading
4. **Feature Extraction**: 80‑channel Mel spectrograms (25 ms windows, 10 ms hop, log compression)
5. **Normalization**: Cepstral mean‑variance normalization per utterance for consistent feature distributions

## Evaluation Results (DER)

| Corpus      | Speakers | DiaPer (%) | EEND‑EDA (%) |
| ----------- | -------- | ---------- | ------------ |
| LibriSpeech | 2        | 1.55       | 1.43         |
| LibriSpeech | 3        | 2.99       | 8.31         |
| LibriSpeech | 4        | 5.56       | 18.73        |
| LibriSpeech | mixed    | 5.70       | 10.50        |
| NeHi        | 2        | 3.28       | 1.50         |
| NeHi        | 3        | 2.02       | 9.68         |
| NeHi        | 4        | 4.05       | 16.17        |
| NeHi        | mixed    | 4.76       | 11.19        |
| VoxCeleb    | 2        | 1.14       | 1.82         |
| VoxCeleb    | 3        | 1.11       | 7.06         |
| VoxCeleb    | 4        | 1.94       | 17.40        |
| VoxCeleb    | mixed    | 2.60       | 8.99         |

*DER: Diarization Error Rate (miss, false alarm, confusion)

## Web Application Overview

- **Backend (Django)**:
  - REST API endpoints: `/api/upload/`, `/api/diarize/`, `/api/transcribe/`
  - Model loading from `backend/models/`, GPU inference support
  - Asynchronous task queue (Celery) for long audio
- **Frontend (React)**:
  - File upload widget, progress bar, interactive speaker timeline
  - Time‑aligned transcripts displayed in Nepali/English
  - Embedded demo video (`media/recording.mp4`) playable inline


## 🎬 Demo Video

Watch our demonstration video below to see the speaker diarization application in action:

<video width="640" height="360" controls>
  <source src="media/demo_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## Project Report
Access project report here:
[Project Report](media/Major_Project_Final.pdf)
