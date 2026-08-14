# Emotion Detection in Text using Bidirectional LSTM

A deep learning project for multi-class text emotion classification built with **TensorFlow / Keras** and trained on the **`dair-ai/emotion`** dataset from Hugging Face.

---

## 📌 Project Overview

This project builds, trains, and serializes a **Bidirectional LSTM** neural network to detect six primary emotions from input text:
* **0:** Sadness
* **1:** Joy
* **2:** Love
* **3:** Anger
* **4:** Fear
* **5:** Surprise

The notebook covers the complete machine learning workflow—from data loading and sequence preprocessing to model architecture design, evaluation, serialization (`.keras` format), and round-trip verification using loaded model instances.

---

## 🧠 Model Architecture

The model is built using TensorFlow's `Sequential` API:

```text
Input (shape: [100])
   │
Embedding (vocab_size: 10,000, output_dim: 128)
   │
Dropout (rate: 0.5)
   │
Bidirectional LSTM (units: 64, return_sequences: False)
   │
Dense (units: 32, activation: 'relu')
   │
Dense (units: 6, activation: 'softmax')
```

### Hyperparameters:
* Vocabulary Size: 10,000 words (Out-Of-Vocabulary Token: <UNK>)
* Maximum Sequence Length: 100 tokens
* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Batch Size: 64
* Epochs: 5

## 🛠 Setup & Installation
1. Clone the repository:

```Bash
git clone https://github.com/Nazmaa-17/MLProject_EmotionDetection.git
cd MLProject_EmotionDetection
```
2. Install dependencies:

```Bash
pip install tensorflow datasets numpy
```

## 🚀 How to Run
### Option 1: Run in Google Colab
Click the Open in Colab badge at the top of the notebook inside GitHub to launch and execute the project directly in Google Colab.

### Option 2: Run Locally
Open Jupyter Notebook or JupyterLab:

```Bash
jupyter notebook emotion_detection.ipynb
```
## 📂 Project Workflow
1. Data Acquisition: Loaded dair-ai/emotion dataset split into train, validation, and test sets.

2. Preprocessing: Tokenized text into sequences, applied padding (max_length=100), and one-hot encoded multi-class labels.

3. Training & Evaluation: Model trained over 5 epochs and evaluated on unseen test data.

4. Serialization & Round-trip Verification:


   * Model exported to native Keras format: emotion_classifier_model.keras
   * Model reloaded into memory using tf.keras.models.load_model()
   * Verified inference logic over custom sample texts:

```Python
test_samples = [
    "I am feeling very happy today and everything is going well.",
    "This is making me quite upset and angry.",
    "I am so sad to hear this news, it truly breaks my heart.",
    "This is amazing!",
    "I am so frustrated right now.",
    "I'm a little bit worried about what will happen next."
]
```

## 📊 Sample Inference Output

```Plaintext
Model saved successfully to 'emotion_classifier_model.keras'
Model loaded successfully!

--- Testing Loaded Model Predictions ---
Text: 'I am feeling very happy today and everything is going well.' -> Predicted Emotion: Joy
Text: 'This is making me quite upset and angry.' -> Predicted Emotion: Anger
Text: 'I am so sad to hear this news, it truly breaks my heart.' -> Predicted Emotion: Sadness
Text: 'This is amazing!' -> Predicted Emotion: Joy
Text: 'I am so frustrated right now.' -> Predicted Emotion: Anger
Text: 'I'm a little bit worried about what will happen next.' -> Predicted Emotion: Fear
```

## 📄 License
Distributed under the MIT License. See LICENSE for details.
