# Speaker Diarization in Nepali Language

This project implements end-to-end Speaker Diarization Nepali (and related) languages, using the following two models:

- **EEND‑EDA**: Encoder–decoder attractor networks with self‑attention for handling unknown speaker counts.
- **DiaPer**: Perceiver‑based attractor networks offering  better performance, more accurate speaker count estimation, and faster inference time.

The system works well with 2 to 4 speakers, even in mixed situations, reaching less than 5% DER on Nepali test sets and performing nearly as well as the best systems on English datasets.

## Reference Implementations
In our work, we used the model architecture and modify the necessary configurations from the official GitHub repositories: [BUTSpeechFIT/EEND](https://github.com/BUTSpeechFIT/EEND) and [BUTSpeechFIT/DiaPer](https://github.com/BUTSpeechFIT/DiaPer). These references provided a solid foundation for implementing and evaluating our speaker diarization models.

## Dataset Sources

| Dataset                                                                                          | Language | #Speakers |
| ------------------------------------------------------------------------------------             | -------- | --------- |
| [LibriSpeech](https://www.openslr.org/12/)                                                       | English  | 921       |
| [VoxCeleb](https://huggingface.co/datasets/ProgramComputer/voxceleb)                             | English  | 1,211     |
| [Nepali Female Speakers](https://www.openslr.org/43/)                                            | Nepali   | 18        |
| [Hindi Audio](https://github.com/shivam-shukla/Speech-Dataset-in-Hindi-Language/)  | Hindi    | 100       |

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

# Web Application Setup

## 📋 Prerequisites
- Python 3.8+ installed
- Node.js 14+ and npm installed
- Git (optional, for cloning repositories)

## 🧠 Backend Setup (Django)

Navigate to backend directory:
```bash
cd backend
```

Create a virtual environment:
```bash
python -m venv venv
```

Activate the virtual environment:
```bash
source venv/bin/activate    # Linux/macOS
```
```bash
venv\Scripts\activate   # Windows
```

Install required Python packages:
```bash
pip install -r requirements.txt
```

Run the Django development server:
```bash
python manage.py runserver
```

> ⚠️ **Important**: Ensure your trained diarization models are placed in the respective directory.

## Frontend Setup (React)

Navigate to frontend directory:
```bash
cd frontend
```

Install required npm packages:
```bash
npm install
```

Start the React development server:
```bash
npm start
```


## 🎬 Demo Video

Watch our demonstration video below to see the speaker diarization application in action:
[▶️ Demo Video](media/demo_video.mp4)

<p align="center">
  <img src="media/demo_preview.gif" width="100%" />
</p>


## Project Report
Access project report here:
[Project Report](media/Major_Project_Final.pdf)

