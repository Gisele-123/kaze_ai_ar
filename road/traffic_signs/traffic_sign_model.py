import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import cv2

IMG_SIZE=(64,64)
DATA_DIR="DATA"
TEST_DIR="TEST"
LABELS_FILE="labels.csv"
print("All files are found")
labels_df=pd.read_csv(LABELS_FILE)
class_labels= {row["ClassId"]: row["Name"] for _, row in labels_df.iterrows()}
print("labels.csv is read for now")
image_paths=[]
image_labels=[]

for class_id in os.listdir(DATA_DIR):
    class_path=os.path.join(DATA_DIR, class_id)
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_file))
            image_labels.append(int(class_id))

images=np.array([cv2.resize(cv2.imread(img), IMG_SIZE).flatten() for img in image_paths])
labels=np.array(image_labels)

X_train, X_val, y_train, y_val=train_test_split(images,labels, test_size=0.3, random_state=42)

encoder=LabelBinarizer()
y_train=encoder.fit_transform(y_train)
y_val=encoder.transform(y_val)

model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, np.argmax(y_train, axis=1))

y_pred=model.predict(X_val)
accuracy=accuracy_score(np.argmax(y_val, axis=1), y_pred)
print(f"Validation accuracy: {accuracy*100:.2f}%")

import joblib
joblib.dump(model, "traffic_sign_model.pkl")

def predict_test_images(model, test_dir):
    for img_file in os.listdir(test_dir):
        img_path=os.path.join(test_dir, img_file)
        img=cv2.resize(cv2.imread(img_path), IMG_SIZE).flatten().reshape(1, -1)
        prediction=model.predict(img)
        print(f"{img_file}: {class_labels.get(prediction[0], "Unknown")}")

predict_test_images(model, TEST_DIR)