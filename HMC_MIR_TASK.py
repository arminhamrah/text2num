#HMC MIR Coding Task
#Armin Hamrah
#Nov 30, 2024 (10am-12pm)

#1 - Data Prep
import inflect
print("Inflect is installed and working!")
import random

#generate dataset
def gen_synth_data():
    p = inflect.engine()
    data = [(p.number_to_words(i).replace('-', ' ').lower(), i) for i in range(10000)]
    random.shuffle(data)
    return data

#split data -> train, validation, and test sets
def split_data(data, train_ratio=0.8, validation_ratio=0.1): #let's use 80% to train, 10% to validate, and 10% to test the model
    train_size = int(len(data) * train_ratio)
    validation_size = int(len(data) * validation_ratio)
    train_data = data[:train_size]
    validation_data = data[train_size:train_size + validation_size]
    test_data = data[train_size + validation_size:]
    return train_data, validation_data, test_data

data = gen_synth_data()
train_data, validation_data, test_data = split_data(data)

#test
print(f"Train sample: {train_data[:3]}")

#2 - Preprocessing
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

#tokenize text
def preprocess_data(data, tokenizer=None, max_len=10):
    texts, numbers = zip(*data)
    if tokenizer is None:
        tokenizer = Tokenizer(lower=True, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    # Convert numbers to digits
    numbers = [[int(d) for d in f"{num:04}"] for num in numbers] #converting textual numbers into arrays
    numbers = np.array(numbers)

    return padded_sequences, numbers, tokenizer

#preprocess the 3 datasets
max_len = 4
X_train, y_train, tokenizer = preprocess_data(train_data, max_len=max_len)
X_validation, y_validation, _ = preprocess_data(validation_data, tokenizer, max_len=max_len)
X_test, y_test, _ = preprocess_data(test_data, tokenizer, max_len=max_len)

#test
print(f"Sample tokenized text: {X_train[:3]}")
print(f"Sample output numbers: {y_train[:3]}")

#3 - Model Construction
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, TimeDistributed

#defining model
def build_model(vocab_size, max_len, output_size=4, embedding_dim=64, rnn_units=128):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        GRU(units=rnn_units, return_sequences=True),
        TimeDistributed(Dense(10, activation='softmax'))
    ])
    model.build(input_shape=(None, max_len)) #added to fill table
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#building model
vocab_size = len(tokenizer.word_index) + 1  # Add 1 for <OOV>
model = build_model(vocab_size, max_len)

#printing model summary
model.summary()

#4 - Train Model
#reshape outputs to match loss function input
y_train_reshaped = np.expand_dims(y_train, axis=-1)
y_validation_reshaped = np.expand_dims(y_validation, axis=-1)

#actually train model
history = model.fit(
    X_train, y_train_reshaped,
    validation_data=(X_validation, y_validation_reshaped),
    epochs=5, #at 5 epochs, accuracy of 0.5820 & loss of 1.1150, but making epochs 20 makes accuracy 0.5947 and loss 1.0520 (pretty marginal)
    batch_size=64
)

#evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, np.expand_dims(y_test, axis=-1))
print(f"Test Accuracy: {test_accuracy:.2f}")

#5 - Test Model
def predict_number(text_input):
    sequence = tokenizer.texts_to_sequences([text_input])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_digits = np.argmax(prediction, axis=-1)[0]
    return int("".join(map(str, predicted_digits)))

#test run
test_input = "one thousand twenty seven"
predicted_output = predict_number(test_input)
print(f"Input: {test_input}, Predicted Output: {predicted_output}")