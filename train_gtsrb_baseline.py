"""
Traffic Sign Recognition
Works with: train.pickle, valid.pickle, test.pickle, label_names.csv
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.metrics import classification_report, confusion_matrix
import json

# -----------------------------
# 1. Config
# -----------------------------
DATA_DIR = "./"   # folder containing train.pickle, valid.pickle, test.pickle
CHECKPOINT_PATH = "checkpoints/cp.weights.h5"
LAST_EPOCH_FILE = "checkpoints/last_epoch.txt"
BATCH_SIZE = 32   # safe for CPU
EPOCHS = 30
IMG_HEIGHT, IMG_WIDTH = 32, 32
NUM_CLASSES = 43  # GTSRB

os.makedirs("checkpoints", exist_ok=True)

# -----------------------------
# 2. Load Dataset
# -----------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

train_data = load_pickle(os.path.join(DATA_DIR, "data/train.pickle"))
valid_data = load_pickle(os.path.join(DATA_DIR, "data/valid.pickle"))
test_data  = load_pickle(os.path.join(DATA_DIR, "data/test.pickle"))

X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = valid_data['features'], valid_data['labels']
X_test, y_test = test_data['features'], test_data['labels']

# Normalize to [0,1]
X_train, X_val, X_test = X_train/255.0, X_val/255.0, X_test/255.0

# If grayscale (H,W), convert to RGB
def ensure_rgb(X):
    if X.ndim == 3:
        X = np.stack([X]*3, axis=-1)
    if X.shape[-1] == 1:
        X = np.concatenate([X]*3, axis=-1)
    return X

X_train, X_val, X_test = ensure_rgb(X_train), ensure_rgb(X_val), ensure_rgb(X_test)

print("Shapes - Train:", X_train.shape, "Valid:", X_val.shape, "Test:", X_test.shape)

# -----------------------------
# 3. Load label names (optional)
# -----------------------------
label_map = {}
label_csv = os.path.join(DATA_DIR, "label_names.csv")
if os.path.exists(label_csv):
    import pandas as pd
    df = pd.read_csv(label_csv)
    if {"ClassId","SignName"}.issubset(df.columns):
        label_map = {int(r["ClassId"]): r["SignName"] for _, r in df.iterrows()}
    else:
        label_map = {int(r.iloc[0]): str(r.iloc[1]) for _, r in df.iterrows()}
    print(f"Loaded {len(label_map)} label names")

# -----------------------------
# 4. Build Model
# -----------------------------
def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH,3)),
        layers.Conv2D(32,3,activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128,3,activation="relu"),
        layers.Flatten(),
        layers.Dense(128,activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES,activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

model = build_model()
model.summary()

# -----------------------------
# 5. Resume from checkpoint
# -----------------------------
initial_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print("✅ Found checkpoint, loading weights...")
    model.load_weights(CHECKPOINT_PATH)
    if os.path.exists(LAST_EPOCH_FILE):
        with open(LAST_EPOCH_FILE,"r") as f:
            initial_epoch = int(f.read().strip())
    print(f"Resuming from epoch {initial_epoch+1}")

# -----------------------------
# 6. Callbacks
# -----------------------------
checkpoint_cb = ModelCheckpoint(
    CHECKPOINT_PATH, save_weights_only=True, verbose=1
)

class EpochSaver(Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open(LAST_EPOCH_FILE,"w") as f:
            f.write(str(epoch))

earlystop_cb = EarlyStopping(patience=5, restore_best_weights=True, verbose=1)

# -----------------------------
# 7. Train
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=[checkpoint_cb, EpochSaver(), earlystop_cb],
    verbose=2
)

# -----------------------------
# 8. Plot training curves
# -----------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend(); plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend(); plt.title("Accuracy")
plt.savefig("training_curves.png")
print("Saved training curves: training_curves.png")

# -----------------------------
# 9. Evaluate on test set
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")

# Predictions + report
y_pred = np.argmax(model.predict(X_test, batch_size=BATCH_SIZE), axis=1)
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("confusion_matrix.png")
print("Saved confusion matrix: confusion_matrix.png")

# -----------------------------
# 10. Save final model + labels
# -----------------------------
model.save("gtsrb_baseline_final.h5")
with open("label_map.json","w") as f:
    json.dump(label_map if label_map else {i:str(i) for i in range(NUM_CLASSES)}, f, indent=2)
print("Saved final model and label_map.json")
