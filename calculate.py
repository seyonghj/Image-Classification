import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "garbage_model.h5"
TEST_DIR = "C:/Users/markl/PYTHON/Image Classification/Garbage classification/Garbage classification"
IMG_SIZE = 128
BATCH_SIZE = 32

# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model(MODEL_PATH)

# ===============================
# LOAD TEST DATA
# ===============================
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ===============================
# EVALUATE MODEL
# ===============================
loss, accuracy = model.evaluate(test_generator)

print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"❌ Test Loss: {loss:.4f}")