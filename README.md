# LSTM Language Model with Subword Units Input Representations

This are implementations of various LSTM-based language models using Tensorflow. 
Codes are based on tensorflow tutorial on building a PTB LSTM model. 
Some extensions are made to handle input from subword units level, i.e. characters, character ngrams, morpheme segments 
(i.e. from BPE/Morfessor).

## Dependencies
1. Tensorflow (tested on v0.10.0)
2. Python 3

## Training
Use the script train.py to train a model. Below is an example to train a character bi-LSTM model for English.
```
python3 train.py --train_file=data/multi/en/train.txt \
					--dev_file=data/multi/en/dev.txt \
					--save_dir=model \
					--unit=char \
					--composition=bi-lstm \
					--rnn_size=200 \
					--batch_size=32 \
					--num_steps=20 \
					--learning_rate=1.0 \
					--decay_rate=0.5 \
					--keep_prob=0.5 \
					--lowercase \
					--SOS=true
```
Options for units are: **char**, **char-ngram**, **morpheme** (BPE/Morfessor), **oracle**, and **word**.  
Options for compositions are: **none** (word only), **bi-lstm**, and **addition**.

The **morpheme** representation uses BPE-like representation. Each word is replaced by its word segments, for example `imperfect` is written as `im@@perfect`, where `@@` denotes the segment boundary. You can use the segmentation tool provided in [here](http://www.aclweb.org/anthology/P16-1162) to preprocess your dataset.

In the **oracle** setting, you need to replace each word in the data with its morphological analysis. For example, in Czech the word `Dodavatel` is replaced by the following (note that the actual word form is not used for experiment):
```
word:Dodavatel+lemma:dodavatel+pos:NOUN+Animacy:Anim+Case:Nom+Gender:Masc+Negative:Pos+Number:Sing
```
Please look at train.py for more hyperparameter options.

## Testing
To test a model, run test.py.
```
python3 test.py --test_file=data/multi/en/test.txt \
					--save_dir=model
```

## Notes
Character-based bi-LSTM model:  
"Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation".  
http://www.cs.cmu.edu/~lingwang/papers/emnlp2015.pdf

Word segments (BPE) model:  
"Neural Machine Translation of Rare Words with Subword Units"  
http://www.aclweb.org/anthology/P16-1162

Character ngrams:  
The model first segments word into its character ngrams, e.g. cat = ('c', 'a', 't', '^c', 'ca', 'at', 't$). The embedding of the word is computed by summing up all the ngrams embeddings of the word.



