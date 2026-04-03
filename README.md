# 🤟 ASL Real-Time Sign Language Detector

A real-time American Sign Language (ASL) letter recognition system built with a custom CNN trained on the Sign Language MNIST dataset, with live webcam inference using OpenCV.

---

## Demo

> Point your hand at the camera, hold a sign inside the green box, and the model predicts the letter in real time.

![Demo placeholder](https://via.placeholder.com/800x400?text=Add+a+demo+GIF+here)

---

## Features

- 🧠 Custom CNN trained from scratch on Sign Language MNIST (~96% validation accuracy)
- 📷 Live webcam inference with OpenCV
- 🔤 Recognizes **24 ASL letters** (A–Y, excluding J and Z which require motion)
- 🗳️ Majority-vote buffer for stable, flicker-free predictions
- 🪲 Debug view showing the thresholded image and exact model input

---

## Project Structure

```
├── nn.ipynb          # Model training notebook
├── webtest.py        # Live webcam inference script
├── asl_cnn_model.keras  # Saved model (not included — train it yourself, see below)
└── README.md
```

---

## Model Architecture

Built with TensorFlow/Keras:

```
Input: (28, 28, 1) grayscale image

Conv2D(16, 3x3, relu) → MaxPooling
Conv2D(32, 3x3, relu) → MaxPooling
Conv2D(64, 3x3, relu) → MaxPooling
Flatten
Dense(128, relu)
Dense(24, softmax)
```

Trained for 5 epochs with data augmentation (rotation ±10°).  
Final validation accuracy: **~96.6%**

---

## Dataset

[Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist) — 28×28 grayscale images of ASL hand signs, one per letter (24 classes, J and Z excluded).

Download `train.csv` and `valid.csv` from Kaggle and place them in the project root before training.

---

## Getting Started

### 1. Install dependencies

```bash
pip install tensorflow opencv-python numpy
```

### 2. Train the model

Open and run `nn.ipynb` end-to-end. This will save `asl_cnn_model.keras` in the project folder.

### 3. Run live inference

```bash
python webtest.py
```

Update the `MODEL_PATH` variable at the top of `webtest.py` to point to your saved model if needed.

---

## Usage Tips

For best prediction accuracy:

- ✅ Use a **plain, light background** (white wall or desk)
- ✅ Make sure your hand is **well-lit**
- ✅ Fill most of the **green box** with your hand
- ✅ Hold each sign **still** for a moment — the smoothing buffer kicks in after ~15 frames
- ❌ Avoid cluttered or dark backgrounds

---

## How Inference Works

The webcam script preprocesses each frame to match the training data format:

1. Crop the region of interest (green box)
2. Convert to grayscale
3. Apply **Otsu thresholding** to isolate the hand
4. Find the largest contour, crop tightly, pad to square
5. Resize to **28×28** and normalise to `[0, 1]`
6. Feed into the CNN and apply a **15-frame majority vote** for stability

---

## Requirements

| Package | Version |
|---|---|
| Python | 3.9+ |
| TensorFlow | 2.x |
| OpenCV | 4.x |
| NumPy | 1.x |

---

## License

MIT License — feel free to use, modify, and distribute.

---

## Acknowledgements

- [Sign Language MNIST — Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
