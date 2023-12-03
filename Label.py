import pickle
import numpy as np
from nltk.corpus import ConllCorpusReader
import tensorflow as tf


def load_model_and_tokenizer(model_path, word_tokenizer_path, tag_tokenizer_path):
    model = tf.keras.saving.load_model(model_path)
    with open(word_tokenizer_path, 'rb') as f:
        word_tokenizer = pickle.load(f)
    with open(tag_tokenizer_path, 'rb') as f:
        tag_tokenizer = pickle.load(f)
    return model, word_tokenizer, tag_tokenizer


def predict(model, word_tokenizer, tag_tokenizer, sentences, output_file):
  actual_lengths = [len(sentence) for sentence in sentences]
  sentences_lower = [[word.lower() for word in sentence] for sentence in sentences]
  X = word_tokenizer.texts_to_sequences(sentences_lower)
  X = tf.keras.utils.pad_sequences(X, maxlen=200)
  y_pred = model.predict(X)
  y_pred = np.argmax(y_pred, axis=-1)

  reverse_tag_map = dict(map(reversed, tag_tokenizer.word_index.items()))

  with open(output_file, 'w') as f:
      for original_word_seq, pred_tag_seq, actual_length in zip(sentences, y_pred, actual_lengths):
          # Intercept prediction results based on actual length
          valid_pred_tags = pred_tag_seq[-actual_length:]  # Assume padding first
          for i, word in enumerate(original_word_seq):
              pred_tag = reverse_tag_map.get(valid_pred_tags[i], 'NN').upper()
              f.write(f'{word}\t{pred_tag}\n')
          f.write('\n')


def load_data(file_path):
    corp = ConllCorpusReader('.', file_path, ('words',))
    sents = corp.sents()
    return sents


if __name__ == '__main__':
    predict1_file_path = 'Data/unlabeled test set/in_domain_test_without_label.txt'
    predict2_file_path = 'Data/unlabeled test set/unlabeled_test_test.txt'

    rnn_model_path = 'Model/RNN-model.h5'
    rnn_word_tokenizer_path = 'Model/rnn_word_tokenizer.pkl'
    rnn_tag_tokenizer_path = 'Model/rnn_tag_tokenizer.pkl'

    cnn_model_path = 'Model/CNN-model.h5'
    cnn_word_tokenizer_path = 'Model/cnn_word_tokenizer.pkl'
    cnn_tag_tokenizer_path = 'Model/cnn_tag_tokenizer.pkl'

    sentences = load_data(predict1_file_path)
    rnn_model, rnn_word_tokenizer, rnn_tag_tokenizer = load_model_and_tokenizer(rnn_model_path, rnn_word_tokenizer_path,
                                                                                rnn_tag_tokenizer_path)
    cnn_model, cnn_word_tokenizer, cnn_tag_tokenizer = load_model_and_tokenizer(cnn_model_path, cnn_word_tokenizer_path,
                                                                                cnn_tag_tokenizer_path)
    predict(rnn_model, rnn_word_tokenizer, rnn_tag_tokenizer, sentences, f'Predict/rnn_predict1.txt')
    predict(cnn_model, cnn_word_tokenizer, cnn_tag_tokenizer, sentences, f'Predict/cnn_predict1.txt')

    sentences = load_data(predict2_file_path)
    predict(rnn_model, rnn_word_tokenizer, rnn_tag_tokenizer, sentences, f'Predict/rnn_predict2.txt')
    predict(cnn_model, cnn_word_tokenizer, cnn_tag_tokenizer, sentences, f'Predict/cnn_predict2.txt')

