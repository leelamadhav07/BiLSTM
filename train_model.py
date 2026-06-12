import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional
)

# ----------------------------------
# LOAD DATA
# ----------------------------------

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000
)

# ----------------------------------
# PADDING
# ----------------------------------

max_length = 200

X_train = pad_sequences(
    X_train,
    maxlen=max_length
)

X_test = pad_sequences(
    X_test,
    maxlen=max_length
)

# ----------------------------------
# MODEL
# ----------------------------------

model = Sequential([

    Embedding(
        input_dim=10000,
        output_dim=64,
        input_length=max_length
    ),

    Bidirectional(
        LSTM(64)
    ),

    Dropout(0.3),

    Dense(
        1,
        activation="sigmoid"
    )
])

# ----------------------------------
# COMPILE
# ----------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------------
# TRAIN
# ----------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# ----------------------------------
# EVALUATE
# ----------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(
    "Accuracy:",
    accuracy
)

# ----------------------------------
# SAVE
# ----------------------------------

model.save(
    "models/bilstm_model.keras"
)

print(
    "Model Saved Successfully"
)