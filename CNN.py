import pickle

import numpy as np
from nltk.corpus import ConllCorpusReader
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data(file_path):
    corp = ConllCorpusReader('.', file_path, ('words', 'pos'))
    sents = corp.tagged_sents()
    sentences = []
    pos_tags = []
    for sent in sents:
        words = []
        tags = []
        for word, tag in sent:
            words.append(word)
            tags.append(tag)
        sentences.append(words)
        pos_tags.append(tags)
    return sentences, pos_tags


def prepare_data(sentences, pos_tags):
    word_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>') 
    word_tokenizer.fit_on_texts(sentences)
    X = word_tokenizer.texts_to_sequences(sentences)

    tag_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tag_tokenizer.fit_on_texts(pos_tags)
    y = tag_tokenizer.texts_to_sequences(pos_tags)

    X = tf.keras.utils.pad_sequences(X, maxlen=200)
    y = tf.keras.utils.pad_sequences(y, maxlen=200)

    return X, y, word_tokenizer, tag_tokenizer


def load_glove_model(glove_file):
    print("Loading Glove Model")
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print("Loaded Glove Model")
    return embeddings_index


def create_embedding_matrix(word_tokenizer, embeddings_index, embedding_dim):
    vocab_size = len(word_tokenizer.word_index) + 1
    print(vocab_size)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def create_model(X, embedding_matrix, embedding_dim, tag_tokenizer):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=len(embedding_matrix), output_dim=embedding_dim,
                        weights=[embedding_matrix], input_length=X.shape[1], trainable=True))
    model.add(tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(tag_tokenizer.word_index) + 1, activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model


def train_model(model, X_train, y_train):
    checkpoint = tf.keras.callbacks.ModelCheckpoint('Model/CNN-model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    history = model.fit(X_train, np.expand_dims(y_train, -1), batch_size=32, epochs=30, validation_split=0.3,
              callbacks=callbacks_list)
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('CNN model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Model/cnn-train.png')
    plt.show()


def evaluate_model(X_test, y_test, word_tokenizer, tag_tokenizer):
    model = tf.keras.saving.load_model('Model/CNN-model.h5')
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)
    # Reverse tokenization to get original words and POS tags
    reverse_word_map = dict(map(reversed, word_tokenizer.word_index.items()))
    reverse_tag_map = dict(map(reversed, tag_tokenizer.word_index.items()))
    with open('Model/Train_output/cnn-train.txt', 'w') as f:
        flattened_y_pred = []
        flattened_y_true = []
        for word_seq, pred_tag_seq, true_tag_seq in zip(X_test, y_pred, y_test):
            words = [reverse_word_map.get(word) for word in word_seq if word != 0]
            pred_tags = [reverse_tag_map.get(tag) for tag in pred_tag_seq if tag != 0]
            true_tags = [reverse_tag_map.get(tag) for tag in true_tag_seq if tag != 0]
            for word, pred, true in zip(words, pred_tags, true_tags):
                f.write(f'{word} {pred} {true}\n')
                flattened_y_pred.append(pred)
                flattened_y_true.append(true)
            f.write('\n')
    print(classification_report(flattened_y_true, flattened_y_pred))


if __name__ == '__main__':
    # load data
    sentences, pos_tags = load_data('Data/labeled train set/train1.txt')
    sentences2, pos_tags2 = load_data('Data/labeled train set/train2.txt')
    # merge data
    sentences.extend(sentences2)
    pos_tags.extend(pos_tags2)
    # preprocess
    X, y, word_tokenizer, tag_tokenizer = prepare_data(sentences, pos_tags)
    # load GloVe
    embeddings_index = load_glove_model('Glove/glove.twitter.27B.200d.txt')
    embedding_dim = 200  # GloVe dimension
    embedding_matrix = create_embedding_matrix(word_tokenizer, embeddings_index, embedding_dim)
    # build model
    model = create_model(X, embedding_matrix, embedding_dim, tag_tokenizer)
    # train model
    train_model(model, X, y)
    # evaluate model
    evaluate_model(X, y, word_tokenizer, tag_tokenizer)
    # store tokenizer
    with open('Model/cnn_word_tokenizer.pkl', 'wb') as f:
        pickle.dump(word_tokenizer, f)
    with open('Model/cnn_tag_tokenizer.pkl', 'wb') as f:
        pickle.dump(tag_tokenizer, f)
