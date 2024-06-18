import re
import os
import nltk
import logging
import random
import torch
from utils import logging_config
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from datasets import load_dataset
from torchtext.vocab import Vectors

plt.style.use('ggplot')
nltk.download('wordnet')
nltk.download('omw-1.4')

stemmer = WordNetLemmatizer()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)


def custom_standardization(text):
    text = re.sub('<br />', ' ', str(text))
    text = re.sub(r'\W', ' ', str(text))
    # remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text)
    # converting to Lowercase
    text = text.lower()
    # lemmatization
    text = text.split()
    text = [stemmer.lemmatize(word) for word in text]
    text = ' '.join(text)

    return text


def get_embeddings_index(path):
    print(f"Load from {path}")
    embeddings_index = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    return embeddings_index


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


results = []
for data in ['ag_news', 'imdb', 'sst2']:
    for pretrained_vector in ['wiki-news-300d-1M.vec', 'glove.6B.300d.txt', 'random']:
        for eps in ['0.01', '0.1', '1.0', '10.0']:
            for method in ['gau', 'lap', 'trlap', 'mdp', 'maha', 'privemb']:
                accuracies = []
                for seed in [42, 43, 44, 45, 46]:
                    X, y = [], []
                    dataset = load_dataset(path=data, cache_dir='./data')
                    for row in dataset['train']:
                        X.append(custom_standardization(row['sentence'].strip()))
                        y.append(row['label'])
                    for row in dataset['validation']:
                        X.append(custom_standardization(row['sentence'].strip()))
                        y.append(row['label'])
                    X = np.array(X)
                    y = np.array(y)

                    num_classes = len(dataset['train'].features['label'].names)
                    max_features = 10000
                    results = Counter()
                    df = pd.DataFrame(X, columns=['text'])
                    print(df.head())
                    df['text'].str.split().apply(results.update)
                    vocabulary = [key[0] for key in results.most_common(max_features)]

                    tokenizer = Tokenizer(num_words=max_features)
                    tokenizer.fit_on_texts(vocabulary)

                    X = tokenizer.texts_to_sequences(df['text'])
                    X = pad_sequences(X)

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                    print(len(X_train), len(X_val), len(X_test), len(y_train), len(y_val), len(y_test))
                    max_input_lenght = X.shape[1]

                    word_index = dict(zip(vocabulary, range(len(vocabulary))))
                    num_tokens = len(vocabulary) + 2
                    embedding_dim = 300

                    now_train = f'{data}_{method}_{eps}_{pretrained_vector}_{seed}'
                    logging_config('checkpoints')
                    logging.info(now_train)
                    path = f"./checkpoints/{now_train}/fine_tune.txt"
                    setup_seed(seed)
                    if not os.path.exists(path):
                        results.append({
                            'data': data,
                            'method': method,
                            'eps': eps,
                            'pretrained_vector': pretrained_vector,
                            'accuracy': 0,
                            'seed': seed
                        })
                        continue
                    embeddings_index = get_embeddings_index(path)
                    logging.info("Found %s word vectors." % len(embeddings_index))

                    hits, misses = 0, 0
                    # Prepare embedding matrix
                    embedding_matrix = np.zeros((num_tokens, embedding_dim))
                    for word, i in word_index.items():
                        embedding_vector = embeddings_index.get(word)
                        if embedding_vector is not None:
                            embedding_matrix[i] = embedding_vector
                            hits += 1
                        else:
                            misses += 1
                    logging.info("Converted %d words (%d misses)" % (hits, misses))

                    # ----- Define model ----- #
                    model = tf.keras.Sequential()
                    model.add(tf.keras.layers.Embedding(
                        num_tokens,
                        embedding_dim,
                        input_length=max_input_lenght,
                        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                    ))
                    model.add(tf.keras.layers.Dropout(0.3))
                    model.add(tf.keras.layers.GlobalAveragePooling1D())
                    model.add(tf.keras.layers.Dense(16, activation='relu'))
                    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
                    # ----- Compile model ----- #
                    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                  optimizer=tf.keras.optimizers.Adam(1e-4),
                                  metrics=["accuracy"])
                    model.summary()

                    history = model.fit(X_train, y_train, batch_size=256, epochs=50, validation_data=(X_val, y_val))

                    probabilities = model.predict(X_test)
                    pred = np.argmax(probabilities, axis=1)
                    accuracy = accuracy_score(y_test, pred)
                    logging.info(f'Accuracy: {accuracy:.4f}')
                    accuracies.append(accuracy)
                    logging.info(classification_report(y_test, pred))
                    results.append({
                        'data': data,
                        'method': method,
                        'eps': eps,
                        'pretrained_vector': pretrained_vector,
                        'accuracy': accuracy,
                        'seed': seed
                    })
                    # plot_history(history)

                mean_accuracy = np.mean(accuracies)
                logging.info(f"{data}_{method}_{eps}_{pretrained_vector}: {mean_accuracy}")
                results.append({
                    'data': data,
                    'method': method,
                    'eps': eps,
                    'pretrained_vector': pretrained_vector,
                    'accuracy': mean_accuracy,
                    'seed': 0
                })

df = pd.DataFrame(results)
df.to_excel('utility_test_results.xlsx', index=False)
