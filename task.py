import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import re
from nltk.stem import WordNetLemmatizer

df = pd.read_csv("imdb.csv", sep=',')

# ----- Get labels -----
y = np.int32(df.sentiment.astype('category').cat.codes.to_numpy())
# ----- Get number of classes -----
num_classes = np.unique(y).shape[0]

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
    pass


import nltk
nltk.download('wordnet')

df['Cleaned_Text'] = df.review.apply(custom_standardization)


# ----- Prepare text for embedding -----
max_features = 10000


# ----- Get top 10000 most occuring words in list-----
results = Counter()
df['Cleaned_Text'].str.split().apply(results.update)
vocabulary = [key[0] for key in results.most_common(max_features)]

# ----- Create tokenizer based on your top 10000 words -----
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(vocabulary)


# ----- Convert words to ints and pad -----
X = tokenizer.texts_to_sequences(df['Cleaned_Text'].values)
X = pad_sequences(X)


# ----- Split into Train, Test, Validation sets -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


output_dim = 16
max_input_lenght = X.shape[1]


path_to_glove_file = "./checkpoints/yahoo/gau_e_1/fine_tune.txt"

def pri_embedding(path):

    path_to_glove_file = path

    embeddings_index = {}
    with open(path_to_glove_file, encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
 
    print("Found %s word vectors." % len(embeddings_index))

    word_index = dict(zip(vocabulary, range(len(vocabulary))))
    num_tokens = len(vocabulary) + 2
    embedding_dim = 300
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix


# ----- Define model -----
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(
    num_tokens,
    embedding_dim,
    input_length = max_input_lenght,
    #trainable = False,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# ----- Compile model -----
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=["accuracy"])


# ----- Train model -----
history_1 = model.fit(X_train, y_train, batch_size=32,epochs=20, validation_data=(X_val, y_val))

# ----- Train model -----
history_2 = model.fit(X_train, y_train, batch_size=32,epochs=20, validation_data=(X_val, y_val))

# ----- Train model -----
history_3 = model.fit(X_train, y_train, batch_size=32,epochs=20, validation_data=(X_val, y_val))


# ----- Train model -----
history_4 = model.fit(X_train, y_train, batch_size=32,epochs=20, validation_data=(X_val, y_val))


# ----- Evaluate model -----
probabilities = model.predict(X_test)
pred = np.argmax(probabilities, axis=1)

print(" ")
print("Results")

accuracy = accuracy_score(y_test, pred)

print('Accuracy: {:.4f}'.format(accuracy))
print(" ")
print(classification_report(y_test, pred))


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

plot_history(history_1)

acc_1 = history_1.history['accuracy']
val_acc_1 = history_1.history['val_accuracy']
acc_2 = history_2.history['accuracy']
val_acc_2 = history_2.history['val_accuracy']
acc_3 = history_3.history['accuracy']
val_acc_3 = history_3.history['val_accuracy']
acc_4 = history_4.history['accuracy']
val_acc_4 = history_4.history['val_accuracy']
# loss_1 = history_1.history['loss']
# val_loss_1 = history_1.history['val_loss']
x = range(1, len(acc_1) + 1)

plt.figure(figsize=(6, 5))
# plt.subplot(1, 2, 1)
plt.plot(x, acc_1, 'b', label='TL_Training acc')
plt.plot(x, val_acc_1, 'r', label='TL_Validation acc')
plt.plot(x, acc_2, 'yellow', label='L_Training acc')
plt.plot(x, val_acc_2, 'green', label='L_Validation acc')
plt.plot(x, acc_3, 'black', label='None_Training acc')
plt.plot(x, val_acc_3, 'orange', label='None_Validation acc')
plt.plot(x, acc_4, 'pink', label='G_Training acc')
plt.plot(x, val_acc_4, 'cyan', label='G_Validation acc')
plt.title('Training and validation accuracy')
plt.grid()
plt.legend()
plt.savefig("plot_4.png", dpi=500,format="png")

