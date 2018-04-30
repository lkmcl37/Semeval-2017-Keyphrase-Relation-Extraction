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
      
# Data
In this project I used the data provided by SemEval 2017, which is a corpus built from ScienceDirect open access publications. It consists of 500 journal articles evenly distributed among the domains Computer Science, Material Sciences and Physics.
The data contains two types of documents: plain text documents with paragraphs, and brat .ann standoff documents with keyphrase annotations for those paragraphs. The training data consists of 350 documents, 50 are kept for development and 100 for testing. 

# Model
The model takes a sentence produced from the preprocessing stage and predicts the relation of the keyphrases pairs. It has the following structure:
1. The input layer turns the words, their positions and entity type features into a embedding vector and concatenates them.
2. The Bi-GRU layer obtain the word-level attention, which helps the model to better determine which parts of the sentence are most influential with respect to the two keyphrases of interest.
3. The sentence-level attention layer builds attention over multiple instances, which helps make full use of all informative sentences.
4. The softmax layer makes predication on relation type of two keyphrases given the output of previous layers.

# References
1. MIT at SemEval-2017 Task 10: Relation Extraction with Convolutional Neural Networks [Lee et al., 2017]
2. ScienceIE - Extracting Keyphrases and Relations from Scientific Publications [Augenstein et al., 2017 ]
3. Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification [Zhou et al.,2016]
4. Neural Relation Extraction with Selective Attention over Instances [Lin et al.,2016].
5. ScienceIE: https://scienceie.github.io/index.html
