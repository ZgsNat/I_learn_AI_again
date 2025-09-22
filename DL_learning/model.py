import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'winemag-data-130k-v2.csv')

# 1. Load dataset
df = pd.read_csv(data_path)
df = df[['description','points']].dropna()

# 2. Tiền xử lý text
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['description'])
sequences = tokenizer.texts_to_sequences(df['description'])
X = pad_sequences(sequences, maxlen=100)

y = df['points'].values  # hoặc df['price']

# 3. Chia tập train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Xây mô hình
model = Sequential([
    Embedding(input_dim=20000, output_dim=128, input_length=100),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)  # Regression output
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# 5. Train
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)

# 6. Đánh giá
loss, mae = model.evaluate(X_test, y_test)
print("Test MAE:", mae)

sample = ["Aromas of blackberry and spice with smooth tannins."]
seq = tokenizer.texts_to_sequences(sample)
padded = pad_sequences(seq, maxlen=100)
print("Predicted points:", model.predict(padded)[0][0])
