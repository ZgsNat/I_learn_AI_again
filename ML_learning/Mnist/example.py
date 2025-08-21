import numpy as np
import cv2
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir,"mnist.npz")

with open(data_path,"rb") as f:
    data = np.load(f)
    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

# cv2.imshow(str(y_train[1110]),cv2.resize(x_train[1110],(280,280)))
# cv2.waitKey(0)

# print(x_train)
# x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_train = np.reshape(x_train,(x_train.shape[0], -1))
x_test = np.reshape(x_test,(x_test.shape[0], -1))

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# print(x_train.shape)
model = RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))