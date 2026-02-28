import cv2
import numpy as np
import tensorflow as tf

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model("garbage_model.h5")

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
    "textile trash",
    "vegitation",
    "food organics"
]

IMG_SIZE = 128

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("🎥 Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ===============================
    # OBJECT DETECTION (ROBUST)
    # ===============================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Largest contour = main object
        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) > 5000:
            x, y, w, h = cv2.boundingRect(c)

            # Draw stable focus box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi = frame[y:y + h, x:x + w]

            if roi.size != 0:
                roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                roi = roi / 255.0
                roi = np.expand_dims(roi, axis=0)

                preds = model.predict(roi, verbose=0)
                class_id = np.argmax(preds)
                confidence = preds[0][class_id] * 100

                label = f"{class_names[class_id]} ({confidence:.1f}%)"

                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Garbage Classification (Stable Box)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()