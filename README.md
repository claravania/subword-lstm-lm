# subword-lstm-lm

LSTM-based language model with subword unit input representations.

This are implementations of various LSTM-based language models using Tensorflow. 
Codes are based on tensorflow tutorial on building a PTB LSTM model. 
Some extensions are made to handle input from subword units level, i.e. characters, character ngrams, morpheme segments 
(i.e. from BPE/Morfessor).

For character-based model, see:
"Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation".
http://www.cs.cmu.edu/~lingwang/papers/emnlp2015.pdf

For BPE segments, see:
"Neural Machine Translation of Rare Words with Subword Units"
http://arxiv.org/abs/1508.07909

For character ngram model:
The model first segments word into its character ngrams, e.g. cat = ('c', 'a', 't', '^c', 'ca', 'at', 't$). The embedding of the word is computed by summing up all the ngrams embeddings of the word.

The LSTM LM can be trained using:
python train.py

NOTE:
- Please look at train.py for hyperparameters.
- For BPE segments, the input should have the following format: imperfect -> im@@perfect, where '@@' denotes the segment boundary.
- Any kind of segmentation can be used, set the parameter encoding="bpe" and preprocess the input to have the same format as described before.



