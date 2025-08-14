import pickle
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# print(model)

test_data = [[3, 75, 76, 29, 3, 26.6, 0.351, 31]]
prediction = model.predict(test_data)
print(prediction)