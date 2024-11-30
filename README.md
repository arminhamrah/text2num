This program was for a 2-hour timed challenge to create a machine learning model which could translate textual numbers into numerical numbers.
For example, if the input to the model is the string “one thousand four hundred and thirty two”, the output of the model is the string “1432”.
It can handle integers from 0 up to 9,999, and was trained from scratch. My approach to generating this ML model took the following steps:

1. Data preparation (used inflect for synthetic data) - split into 80% train, 10% validation, 10% testing
2. Preprocessing - used TensorFlow’s Tokenizer to tokenize textual inputs into sequences of integers (padded to fixed length of 4 because output length won’t exceed 4)
3. Model - as recommended by ChatGPT, I used an embedding layer, GRU layer, and TimeDistributed layer for my sequential model. The embedding layer converts tokens →  dense vector representations, the GRU processes sequences and captures long-term dependencies efficiently, and the TimeDistributed layer outputs probabilities for each digit (0-9) at each position in the 4-digit sequence. 
4. Training - Model trained for 5 epochs with batch size of 64; validation accuracy capped at ~58.2% with a loss of 1.1150
5. Evals & training - Model got a roughly 59% accuracy rate (when moved to 20 epochs) according to my code, but when I manually changed test_input I got weird results. Oftentimes the model would use several of the same digit twice incorrectly (like ‘55’ for ‘seventy five’), though I didn’t have enough time to investigate this

This was a fun task, and I learned a lot. In terms of next steps, I would like to:
1. Try a transformer-based model (textual encoder and a positional decoder) - could probably better capture relationships between input tokens & digit positions due to attention mechanism
2. Using pre-trained word embeddings, perhaps from Word2Vec or a similar models
3. Increasing model complexity (perhaps switching to LSTM architecture, one we experimented with at LMNT a bit), training for significantly more epochs, and further optimizing loss function
