# Semeval-2017-Keyphrase-Relation-Extraction
In this project, I’ve implemented a word-level and sentence-level combined Bidirectional
GRU network to classify pairs of keyphrases in scientific articles into one of three types: (i)
HYPONYM-OF, (ii) SYNONYM-OF, and (iii) NONE.

# Requirements
Python (>=2.7)
TensorFlow (>=r0.11)
NLTK
SpaCy


# Usage
The data used for training and testing is in the ./data/ folder, which includes:
1) the dictionary that map each relation type to id (relation2id.txt)
2) the dictionary that map entity type to id (keyphrase_type2id)
3) training data, dev, test data (/train/, /dev/, /test/)
4) GloVe pre-trained word vectors (emb_vec.txt).

# Preprocessing
To train or test a model, you’ll first need to vectorize the raw data and turn it into
numpy arrays. For your convenience, the numpy array files for train and test set have been
included in the ./vectorized_data/ folder. But you can still preprocess the data on your own
with the following command:

      python prepare_data.py
      
The files will be outputted to ./vectorized_data/ folder.

# Training
To train a model, please run:

      python train.py

The model will be stored in the ./model/ folder. Modify the line of the source code to
change the model name and path.
For your convenience, a pre-trained model is included in the ./model folder.

# Testing
To test a model with a existing model, please run:
      
      python test.py
      
This command line will output the prediction results to ./test_pred/ folder.

Then you can compute the model performances using:
      
      python eval.py ./data/test/ ./test_pred/ keys
