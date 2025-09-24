import tensorflow as tf
import numpy as np

path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)

text = open(path_to_file, 'r', encoding='utf-8').read()
vocab = sorted(set(text))

# Mapping ký tự <-> số
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

seq_length = 100
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Xây dựng model LSTM

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 512  # nhỏ hơn ví dụ trước để train nhanh

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])

model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)


EPOCHS = 20 
model.fit(dataset, epochs=EPOCHS)

model.save("shakespeare_rnn.h5")

# # Dùng batch_size=1 cho inference
# model_infer = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
# model_infer.set_weights(model.get_weights())

# def generate_text(model, start_string, num_generate=300, temperature=0.7):
#     input_eval = [char2idx.get(s, 0) for s in start_string]
#     input_eval = tf.expand_dims(input_eval, 0)

#     text_generated = []
#     model.reset_states()

#     for _ in range(num_generate):
#         predictions = model(input_eval)
#         predictions = tf.squeeze(predictions, 0) / temperature
#         predicted_id = tf.random.categorical(
#             predictions, num_samples=1)[-1, 0].numpy()

#         input_eval = tf.expand_dims([predicted_id], 0)
#         text_generated.append(idx2char[predicted_id])

#     return start_string + ''.join(text_generated)


# if __name__ == "__main__":
#     start_string = input("Nhập 3 từ bắt đầu: ")
#     print("\n=== Văn bản sinh ra ===\n")
#     print(generate_text(model_infer, start_string))
