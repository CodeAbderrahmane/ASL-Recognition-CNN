
import cv2
import numpy as np
import tensorflow as tf
from collections import Counter

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH          = r"C:\Users\Farma\desktop\gomycode\dlstudy\checkpoint_13\asl_cnn_model.keras"
ASL_LETTERS         = list("ABCDEFGHIKLMNOPQRSTUVWXY")   # 24 classes (no J, Z)
CONFIDENCE_THRESH   = 0.45
BUFFER_SIZE         = 15          # frames for majority-vote smoothing
ROI_X, ROI_Y        = 300, 80     # top-left corner of the capture box
ROI_SIZE            = 300         # box is always square
# ─────────────────────────────────────────────────────────────────────────────

print("Loading model ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Ready.  Press  Q  to quit.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

pred_buffer = []


def extract_hand_roi(gray_roi):
    """
    Given a square grayscale ROI, isolate the hand and return a
    tightly-cropped, centred 28x28 patch -- same style as Sign MNIST.

    Steps
    -----
    1. Adaptive threshold  ->  binary hand mask
    2. Find the largest contour (= hand)
    3. Crop to its bounding box, pad to square, resize to 28x28
    4. Fall back to a plain resize if no contour is found
    """
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # Otsu works well when hand vs background have distinct grey levels
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)

        # ignore tiny noise (< 2% of ROI area)
        if area > 0.02 * gray_roi.shape[0] * gray_roi.shape[1]:
            x, y, w, h = cv2.boundingRect(largest)

            # small padding around the bounding box
            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(gray_roi.shape[1] - x, w + 2 * pad)
            h = min(gray_roi.shape[0] - y, h + 2 * pad)

            crop = gray_roi[y:y+h, x:x+w]

            # pad to square so resize does not squash the sign
            side   = max(crop.shape)
            square = np.full((side, side), 255, dtype=np.uint8)  # white bg
            oy     = (side - crop.shape[0]) // 2
            ox     = (side - crop.shape[1]) // 2
            square[oy:oy+crop.shape[0], ox:ox+crop.shape[1]] = crop

            resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
            return resized, thresh

    # fallback: just resize the whole ROI
    resized = cv2.resize(gray_roi, (28, 28), interpolation=cv2.INTER_AREA)
    return resized, thresh


while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed.")
        break

    frame = cv2.flip(frame, 1)
    H, W  = frame.shape[:2]

    # ── capture box ──────────────────────────────────────────────────────────
    rx, ry, rs = ROI_X, ROI_Y, ROI_SIZE
    cv2.rectangle(frame, (rx, ry), (rx+rs, ry+rs), (0, 230, 0), 2)
    cv2.putText(frame,
                "Plain background . hand fills box",
                (rx, ry - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 0), 1)

    roi_bgr  = frame[ry:ry+rs, rx:rx+rs]
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # ── preprocess -> 28x28 ──────────────────────────────────────────────────
    model_img, thresh_debug = extract_hand_roi(roi_gray)
    norm         = model_img.astype(np.float32) / 255.0
    input_tensor = norm.reshape(1, 28, 28, 1)

    # ── predict ───────────────────────────────────────────────────────────────
    probs      = model.predict(input_tensor, verbose=0)[0]
    best_idx   = int(np.argmax(probs))
    confidence = float(probs[best_idx])

    if confidence >= CONFIDENCE_THRESH:
        pred_buffer.append(ASL_LETTERS[best_idx])
        if len(pred_buffer) > BUFFER_SIZE:
            pred_buffer.pop(0)
        letter = Counter(pred_buffer).most_common(1)[0][0]
        color  = (0, 220, 0)
    else:
        letter = "?"
        color  = (0, 80, 255)

    # ── main display ─────────────────────────────────────────────────────────
    cv2.putText(frame, letter, (15, 80),
                cv2.FONT_HERSHEY_DUPLEX, 3.5, color, 5)
    cv2.putText(frame, f"{confidence*100:.0f}%", (15, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    # top-5 list on the right
    top5 = np.argsort(probs)[::-1][:5]
    for rank, idx in enumerate(top5):
        txt   = f"{ASL_LETTERS[idx]}  {probs[idx]*100:4.1f}%"
        c     = (255, 220, 0) if rank == 0 else (160, 160, 160)
        thick = 2 if rank == 0 else 1
        cv2.putText(frame, txt, (W - 175, 35 + rank * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, thick)

    # ── debug thumbnails (bottom-left) ───────────────────────────────────────
    THUMB = 120
    by    = H - THUMB - 20

    def put_thumb(img_gray, col, label):
        thumb = cv2.resize(img_gray, (THUMB, THUMB), interpolation=cv2.INTER_NEAREST)
        bgr   = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        frame[by:by+THUMB, col:col+THUMB] = bgr
        cv2.putText(frame, label, (col + 2, by + THUMB + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 210, 0), 1)

    put_thumb(roi_gray,      0,           "ROI gray")
    put_thumb(thresh_debug,  THUMB + 4,   "threshold")
    put_thumb(model_img,    (THUMB+4)*2,  "model input")

    cv2.imshow("ASL Prediction  [Q = quit]", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()