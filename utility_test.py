import re
import os
import nltk
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


X, y = [], []
dataset = load_dataset(path='sst2', cache_dir='./data')
for row in dataset['train']:
    X.append(custom_standardization(row['sentence'].strip()))
    y.append(row['label'])
for row in dataset['validation']:
    X.append(custom_standardization(row['sentence'].strip()))
    y.append(row['label'])
X = np.array(X)
y = np.array(y)

num_classes = len(dataset['train'].features['label'].names)

# ----- Prepare text for embedding ----- #
max_features = 10000
# ----- Get top 10000 most occuring words in list----- #
results = Counter()
df = pd.DataFrame(X, columns=['text'])
print(df.head())
df['text'].str.split().apply(results.update)
vocabulary = [key[0] for key in results.most_common(max_features)]
# print(vocabulary)

# ----- Create tokenizer based on your top 10000 words ----- #
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(vocabulary)

# ----- Convert words to ints and pad ----- #
X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X)

# ----- Split into Train, Test, Validation sets -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(len(X_train), len(X_val), len(X_test), len(y_train), len(y_val), len(y_test))
max_input_lenght = X.shape[1]

word_index = dict(zip(vocabulary, range(len(vocabulary))))
num_tokens = len(vocabulary) + 2
embedding_dim = 300

results = []
for data in ['ag_news', 'imdb', 'sst2']:
    for pretrained_vector in ['wiki-news-300d-1M.vec', 'glove.6B.300d.txt', 'random']:
        for eps in ['0.01', '0.1', '1.0', '10.0']:
            for method in ['gau', 'lap', 'trlap', 'mdp', 'maha', 'privemb']:
                path = f"./checkpoints/{data}_{method}_{eps}_{pretrained_vector}/fine_tune.txt"
                if not os.path.exists(path):
                    results.append({
                        'data': data,
                        'method': method,
                        'eps': eps,
                        'pretrained_vector': pretrained_vector,
                        'accuracy': 0
                    })
                    continue
                embeddings_index = get_embeddings_index(path)
                print("Found %s word vectors." % len(embeddings_index))

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
                print("Converted %d words (%d misses)" % (hits, misses))

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
                print(f'\n{data}_{method}_{eps}_{pretrained_vector}\nAccuracy: {accuracy:.4f}\n')
                print(classification_report(y_test, pred))

                plot_history(history)

                results.append({
                    'data': data,
                    'method': method,
                    'eps': eps,
                    'pretrained_vector': pretrained_vector,
                    'accuracy': accuracy
                })

data = []
vocab_file = os.path.join('./vocabulary', f'sst2.txt')
f = open(vocab_file, 'r', encoding="utf-8")
while True:
    line = f.readline()
    if not line:
        break
    line = line[:-1]
    data.append(line.split('\t'))
df = pd.DataFrame(data)

# Use pre-trained word embedding
pretrained_vectors = Vectors(name=f'./embeddings/wiki-news-300d-1M.vec')
weight_matrix = pretrained_vectors.get_vecs_by_tokens(list(df[0]))
weight_matrix = weight_matrix[:10002, :]

df = pd.DataFrame(results)
df.to_excel('utility_test_results.xlsx', index=False)
