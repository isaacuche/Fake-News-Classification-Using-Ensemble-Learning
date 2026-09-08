import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    concatenate
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import stopwords
import re


# Ensure stopwords are available
try:
    stop_words = stopwords.words('english')
except LookupError:
    print("Stopwords not found. Please ensure NLTK stopwords are downloaded.")
    stop_words = []


# Load Dataset
try:
    df = pd.read_csv('data/build.csv')  # Ensure dataset has 'text' and 'label' columns

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()


df['text'] = df['text'].astype(str)


# Preprocess text
# Converts text to lowercase.
# Removes stopwords using NLTK
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]

    return ' '.join(text)


df['cleaned_text'] = df['text'].apply(preprocess_text)


# Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_text'])

vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(df['cleaned_text'])

max_len = 100  # Adjust based on your data

X = pad_sequences(sequences, maxlen=max_len)

y = df['label'].values  # Binary labels: 0 for real, 1 for fake


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# CNN + LSTM Hybrid Model
def create_hybrid_model(vocab_size, max_len):

    input_layer = Input(shape=(max_len,))

    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=128,
        input_length=max_len
    )(input_layer)

    # CNN Layers
    cnn_layer = Conv1D(
        128,
        5,
        activation='relu'
    )(embedding_layer)

    cnn_layer = GlobalMaxPooling1D()(cnn_layer)

    # LSTM Layer
    lstm_layer = LSTM(
        128,
        return_sequences=False
    )(embedding_layer)

    # Combine CNN and LSTM
    concatenated = concatenate([
        cnn_layer,
        lstm_layer
    ])

    dense_layer = Dense(
        64,
        activation='relu'
    )(concatenated)

    dropout_layer = Dropout(
        0.5
    )(dense_layer)

    output_layer = Dense(
        1,
        activation='sigmoid'
    )(dropout_layer)

    model = Model(
        inputs=input_layer,
        outputs=output_layer
    )

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


hybrid_model = create_hybrid_model(
    vocab_size,
    max_len
)


# Train Hybrid Model
hybrid_model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)


# Evaluate Hybrid Model
hybrid_predictions = hybrid_model.predict(X_test)

hybrid_predictions = (
    hybrid_predictions > 0.5
).astype(int)

print("Hybrid Model Classification Report:")

print(
    classification_report(
        y_test,
        hybrid_predictions
    )
)


# Ensemble Learning Models

# Feature Extraction from CNN + LSTM (last dense layer)
feature_extractor = Model(
    inputs=hybrid_model.input,
    outputs=hybrid_model.layers[-2].output
)

train_features = feature_extractor.predict(X_train)

test_features = feature_extractor.predict(X_test)


# Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_clf.fit(
    train_features,
    y_train
)

rf_predictions = rf_clf.predict(
    test_features
)


# Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

gb_clf.fit(
    train_features,
    y_train
)

gb_predictions = gb_clf.predict(
    test_features
)


# Evaluate Ensemble Models

print("Random Forest Classification Report:")

print(
    classification_report(
        y_test,
        rf_predictions
    )
)


print("Gradient Boosting Classification Report:")

print(classification_report(y_test, gb_predictions))