# Dual Model Emotion Detection

## Emotion Detection from Text and Speech

This project implements a dual-modal emotion detection system that recognizes a speaker's emotion from:

1. **Text**: Transcribed speech processed through a BERT-based classifier.
2. **Audio**: Raw speech audio analyzed via spectrograms, Mel-frequency cepstral coefficients (MFCCs), and a deep learning pipeline.

---

### Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Environment and Dependencies](#environment-and-dependencies)
* [Preprocessing](#preprocessing)

  * [Text Cleaning](#text-cleaning)
  * [Audio Feature Extraction](#audio-feature-extraction)
* [Model Architectures](#model-architectures)

  * [Text Emotion Classifier](#text-emotion-classifier)
  * [Audio Emotion Classifier](#audio-emotion-classifier)
* [Training and Evaluation](#training-and-evaluation)
* [Project Structure](#project-structure)
* [Results](#results)
* [Future Work](#future-work)
* [License](#license)

---

## Project Overview

This repository contains code to train and evaluate two separate emotion detection models:

* **Text-based Model**: Uses `bert-base-cased` from Hugging Face to classify emotions (`anger`, `fear`, `joy`, `love`, `sadness`, `surprise`) from input text.
* **Audio-based Model**: Uses Wav2Vec2 for speech-to-text transcription and extracts spectrograms/MFCCs to predict emotion categories from raw audio.

Emotions are predicted using a six-class classification with performance metrics and confusion matrices generated for analysis.

## Dataset

The dataset combines `training.csv`, `test.csv`, and `validation.csv` containing labeled utterances. Each row has:

* `text`: The transcribed or raw utterance text.
* `label`: The emotion category.

Audio files from LJSpeech (for demonstration) show the STT pipeline; for full audio-based training, replace with your labeled speech dataset.

## Environment and Dependencies

* Python 3.8+
* TensorFlow 2.x
* Hugging Face Transformers
* text\_hammer
* librosa
* PyTorch (for Wav2Vec2 inference)
* scikit-learn
* seaborn, matplotlib
* pandas, numpy

```bash
pip install -r requirements.txt
```

## Preprocessing

### Text Cleaning

* Lowercasing, contraction expansion (using `text_hammer`).
* Removal of emails, HTML tags, special characters, accented characters.
* Computation of word counts for exploratory analysis.

### Audio Feature Extraction

* Load WAV at 16kHz via `librosa`.
* Use `facebook/wav2vec2-base-960h` for speech-to-text transcription.
* Compute spectrograms and MFCCs (code not shown here, but can be added in `audio_processor.py`).

## Model Architectures

### Text Emotion Classifier

* **Base**: `TFBertModel` with `bert-base-cased` weights.

* **Structure**:

  1. Input IDs and attention mask.
  2. Last hidden state pooled via `GlobalMaxPool1D`.
  3. Dense layers (128, then 32 units) with ReLU and Dropout.
  4. Output layer: Dense(6) with sigmoid activation.

* **Training**:

  * Adam optimizer (lr=5e-5, decay=0.01, clipnorm=1.0).
  * Loss: Categorical Crossentropy.
  * Metrics: Categorical Accuracy.
  * Early stopping on `val_loss` with patience=3.

### Audio Emotion Classifier

* **STT**: `Wav2Vec2ForCTC` for transcription.
* **Feature Extraction**: Spectrogram and MFCC extraction pipeline.
* **Model**: Custom CNN/RNN layers on extracted features (to be implemented).

## Training and Evaluation

1. Split data with stratified sampling (70% train, 30% test).
2. Tokenize text sequences (max length=70).
3. Fit model for up to 50 epochs with batch size 36.
4. Evaluate on test set: accuracy, classification report, confusion matrix.
5. Visualize results with heatmaps.


## Project Structure

```
├── emotion_detection.ipynb
└── README.md
```

## Results

* **Text Model**: Achieved **93.25%** test accuracy.
* **Audio Model**: Achieved **45%** test accuracy.

Refer to `results/` for confusion matrices and training curves.

## Future Work

* Implement end-to-end training for audio model.
* Experiment with transformer-based audio encoders.

## License

This project is licensed under the MIT License.

---

*Feel free to open issues or submit pull requests!*
