## Requirements

+ python 3.10.13
+ tensorflow 2.12.0
+ numpy
+ nltk
+ sklearn
+ matplotlib-3.8.2

## Usage

### To train models

You can replace the training data in the Data/labeled train set. The training data format is one word per line. Words and labels are separated by spaces, and sentences are separated by blank lines.

```shell
python CNN.py
```

```shell
python RNN.py
```

After running, the model file, as well as tag_tokenizer and word_tokenizer will be generated in the Model folder. And evaluate the effect of the model and output the confusion matrix.

### To predict

```shell
python Label.py
```

the prediction results and evaluation results will be generated in the Predict folder.

### To compare answer

```shell
python Compare.py
```

a comparison of the predicted results and the correct results will be output in the console.

## Model Structure

### CNN

```python
def create_model(X, embedding_matrix, embedding_dim, tag_tokenizer):
    model = Sequential()
    model.add(Embedding(input_dim=len(embedding_matrix), output_dim=embedding_dim,
                        weights=[embedding_matrix], input_length=X.shape[1], trainable=False))
    model.add(Conv1D(64, 5, activation='relu', padding='same'))
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(len(tag_tokenizer.word_index) + 1, activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model
```

Our model uses a Sequential structure, combining embedding layers, convolutional layers, Dropout layers and TimeDistributed Dense layers.

- **embedding layer**: Use pre-trained word embeddings.
- **convolutional layer**: Extract local features.
- **Dropout layer**: Prevent overfitting.
- **TimeDistributed Dense layer**: Perform part-of-speech prediction.

### RNN

```python
def create_model(X, embedding_matrix, embedding_dim, tag_tokenizer):
    model = Sequential()
    model.add(Embedding(input_dim=len(embedding_matrix), output_dim=embedding_dim, weights=[embedding_matrix],
                        input_length=X.shape[1], trainable=False))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Dropout(0.3))  
    model.add(Dense(len(tag_tokenizer.word_index) + 1, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001))  
    return model
```

The model uses a bidirectional long short-term memory network (Bi-LSTM) to capture the contextual information in the sentence.

- **embedding layer**: Use pre-trained word embeddings.
- **BiLSTM**:Capture contextual information and improve annotation performance.
- **Dropout layer**: Prevent overfitting.
- **Dense layer**: Use the softmax activation function to output the probability distribution of parts of speech.