#!/usr/bin/python
# Author: Clara Vania

import numpy as np
import os
import re
import sys
import random
import pickle
import collections
import operator


class TextLoader:
    def __init__(self, args, train=True):
        self.save_dir = args.save_dir
        self.batch_size = args.batch_size
        self.num_steps = args.num_steps
        self.out_vocab_size = args.out_vocab_size
        self.unit = args.unit
        self.composition = args.composition
        self.lowercase = args.lowercase
        self.n = 3
        self.use_all_morphemes = True
        self.eos = args.eos
        self.sos = args.sos

        self.words_vocab_file = os.path.join(self.save_dir, "words_vocab.pkl")
        if self.unit == "oracle":
            self.lowercase = True
        if self.unit != "word":
            self.sub_vocab_file = os.path.join(self.save_dir, "sub_vocab.pkl")

        # Variables
        self.max_word_len = 0
        self.subword_vocab_size = 0
        self.word_vocab_size = 0
        print("reading text")
        # Dictionaries and lists
        self.char_to_id = dict()
        self.word_to_id = dict()
        self.unk_word_list = set()
        self.unk_char_list = set()
        if self.unit == "morpheme" or self.unit == "oracle":
            self.max_morph_per_word = 0
            self.morpheme_to_id = dict()
            self.unk_morph_list = set()
        if self.unit == "char-ngram":
            self.max_ngram_per_word = 0
            self.ngram_to_id = dict()
            self.unk_ngram_list = set()

        self.output_vocab = {}
        self.unk_list = set()
        if args.output_vocab_file:
            self.output_vocab, self.unk_list = self.load_vocab_dict(args.output_vocab_file)

        if train:
            self.train_data = self.read_dataset(args.train_file)
            self.dev_data = self.read_dataset(args.dev_file)
            self.preprocess()
        else:
            self.load_preprocessed()

    @staticmethod
    def is_hyperlink(word):
        keywords = ('www', 'http', 'html')
        for key in keywords:
            if key in word:
                return True
        return False

    @staticmethod
    def padding(arr, max_len, vocab):
        """
        Padding a vector of characters or words
        :param arr: array to be padded
        :param start: start symbol, ex: <w> for character sequence
        :param end: end symbol
        :param max_len: maximum length for padding
        :param vocab: vocabulary to get the indexes of start, end, and PAD symbols
        :return:
        """
        tmp = []
        if len(arr) <= max_len:
            tmp.extend([element for element in arr])
            tmp.extend([vocab['<PAD>'] for _ in range(max_len - len(arr))])
        else:
            tmp.extend([element for element in arr[:max_len]])
        return tmp

    def preprocess(self):
        """
        Preprocess dataset and build vocabularies
        """
        self.word_to_id, self.unk_word_list = self.build_vocab(mode="word")
        self.word_vocab_size = len(self.word_to_id)
        self.max_word_len = self.get_max_word_length(self.word_to_id)

        with open(self.words_vocab_file, 'wb') as f:
            pickle.dump((self.word_to_id, self.unk_word_list), f)
        if self.unit != "word":
            self.preprocess_sub_units()

    def preprocess_sub_units(self):
        """
        Build dictionaries for sub word units
        """
        if self.unit == "char":
            self.preprocess_char()
        elif self.unit == "char-ngram":
            self.preprocess_char_ngram()
        elif self.unit == "morpheme":
            self.preprocess_morpheme()
        elif self.unit == "oracle":
            self.preprocess_oracle()
        else:
            sys.exit("Unknown unit")

    def preprocess_char(self):
        """
        Build dictionaries for character representation
        """
        self.char_to_id, self.unk_char_list = self.build_vocab(mode="char")
        self.subword_vocab_size = len(self.char_to_id)
        with open(self.sub_vocab_file, 'wb') as f:
            pickle.dump((self.char_to_id, self.unk_char_list, self.max_word_len), f)

    def preprocess_char_ngram(self):
        """
        Build dictionaries for char-ngram representation
        """
        self.char_to_id, self.unk_char_list = self.build_vocab(mode="char")
        self.ngram_to_id, self.unk_ngram_list, self.max_ngram_per_word = self.build_ngram_vocab(self.n)
        for ch in self.char_to_id:
            if ch not in self.ngram_to_id:
                self.ngram_to_id[ch] = len(self.ngram_to_id)
        self.subword_vocab_size = len(self.ngram_to_id)
        with open(self.sub_vocab_file, 'wb') as f:
            pickle.dump((self.ngram_to_id, self.unk_char_list, self.unk_ngram_list, self.max_ngram_per_word), f)

    def preprocess_morpheme(self):
        """
        Preprocess for morpheme model
        """
        self.char_to_id, self.unk_char_list = self.build_vocab(mode="char")
        self.morpheme_to_id, self.unk_morph_list, self.max_morph_per_word = self.build_morpheme_vocab()
        for ch in self.char_to_id:
            if ch not in self.morpheme_to_id:
                self.morpheme_to_id[ch] = len(self.morpheme_to_id)
        self.subword_vocab_size = len(self.morpheme_to_id)
        with open(self.sub_vocab_file, 'wb') as f:
            pickle.dump((self.morpheme_to_id, self.unk_char_list, self.unk_morph_list, self.max_morph_per_word), f)

    def preprocess_oracle(self):
        """
        Preprocess for morpheme model
        """
        self.morpheme_to_id, self.max_morph_per_word = self.build_oracle_vocab()
        self.subword_vocab_size = len(self.morpheme_to_id)
        with open(self.sub_vocab_file, 'wb') as f:
            pickle.dump((self.morpheme_to_id, self.max_morph_per_word), f)

    def load_preprocessed(self):
        """
        Load preprocessed dictionaries, this is called during testing.
        """
        with open(self.words_vocab_file, 'rb') as f:
            self.word_to_id, self.unk_word_list = pickle.load(f)
            self.word_vocab_size = len(self.word_to_id)

        if self.unit != "word":
            with open(self.sub_vocab_file, 'rb') as f:
                if self.unit == "char":
                    self.max_word_len = self.get_max_word_length(self.word_to_id) + 2
                    self.char_to_id, self.unk_char_list, self.max_word_len = pickle.load(f)
                    self.subword_vocab_size = len(self.char_to_id)
                elif self.unit == "char-ngram":
                    self.ngram_to_id, self.unk_char_list, self.unk_ngram_list, \
                    self.max_ngram_per_word = pickle.load(f)
                    self.subword_vocab_size = len(self.ngram_to_id)
                elif self.unit == "morpheme":
                    self.morpheme_to_id, self.unk_char_list, self.unk_morph_list, \
                    self.max_morph_per_word = pickle.load(f)
                    self.subword_vocab_size = len(self.morpheme_to_id)
                elif self.unit == "oracle":
                    self.morpheme_to_id, self.max_morph_per_word = pickle.load(f)
                    self.subword_vocab_size = len(self.morpheme_to_id)
                else:
                    sys.exit("Unknown unit")

    def load_vocab_dict(self, vocab_file):
        """
        Load preprocessed dictionaries, this is called during testing.
        """
        with open(vocab_file, 'rb') as f:
            output_vocab, unk_list = pickle.load(f)
        return output_vocab, unk_list

    def build_vocab(self, mode):
        """
        Build vocabularies: word_to_id OR char_to_id
        Keys: words or chars
        Values: ids
        unk_list defines the list of words or chars which only occur once in the
        training data.
        :param mode: defines the type of vocabulary: word or char
        """
        if mode == "word":
            data = self.read_words()
        else:
            data = self.read_chars()

        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[-1], x[0]))
        item_to_id = dict()
        unk_list = set()

        if mode == "char":
            # '^' is a start of word symbol
            # '$' is a end of word symbol
            item_to_id = self.add_to_dict(item_to_id, '^', '$')
        else:
            if len(self.output_vocab) > 0:
                print('Using output vocab!')
                print('Output vocab size:', self.out_vocab_size)
                sorted_vocab = sorted(self.output_vocab.items(), key=operator.itemgetter(1))

                for k, v in sorted_vocab:
                    item_to_id[k] = len(item_to_id)
                    if len(item_to_id) == self.out_vocab_size:
                        break
            else:
                item_to_id['<unk>'] = len(item_to_id)
                if self.sos != '':
                    item_to_id[self.sos] = len(item_to_id)
                if self.eos != '':
                    item_to_id[self.eos] = len(item_to_id)

        for i, (token, freq) in enumerate(count_pairs):
            if token not in item_to_id:
                item_to_id[token] = len(item_to_id)
            if freq == 1:
                unk_list.add(token)
        return item_to_id, unk_list

    def build_ngram_vocab(self, n):
        """
        Build a dictionary of character ngrams found in the training data
        Keys: n-grams
        Values: n-gram frequencies
        :param n: length of the character ngram
        :return: a dictionary of character ngrams, (ngram, freq) key-value pairs
        and max_ngram_per_word
        """
        max_ngram_per_word = 0
        ngram_dict = collections.defaultdict(int)
        for word in self.train_data:
            if word == self.eos or word == self.sos:
                continue
            _word = '^' + word + '$'
            ngram_counts = len(_word) - n + 1
            if ngram_counts > max_ngram_per_word:
                max_ngram_per_word = ngram_counts
            for i in range(ngram_counts):
                ngram = _word[i:i + n]
                ngram_dict[ngram] += 1

        unk_ngram_list = set()
        item_to_id = dict()
        sorted_dict = sorted(ngram_dict.items(), key=operator.itemgetter(1), reverse=True)
        for token, freq in sorted_dict:
            if freq == 1:
                unk_ngram_list.add(token)
            if token not in item_to_id:
                item_to_id[token] = len(item_to_id)
        return item_to_id, unk_ngram_list, max_ngram_per_word

    def build_morpheme_vocab(self):
        """
        Build morpheme vocab from a given file
        Keys: morphemes
        Values: morpheme frequencies
        :return: a dictionary: (morpheme, freq) key-value pairs and max_morph_per_word
        """
        max_morph_per_word = 0
        morpheme_dict = collections.defaultdict(int)
        splitter = "@@"
        for token in self.train_data:
            if token == self.eos or token == self.sos:
                continue
            token = '^' + token + '$'
            morphemes = token.split(splitter)
            if len(morphemes) > max_morph_per_word:
                max_morph_per_word = len(morphemes)
            for morpheme in morphemes:
                morpheme_dict[morpheme] += 1

        unk_morpheme_list = set()
        item_to_id = dict()
        sorted_dict = sorted(morpheme_dict.items(), key=operator.itemgetter(1), reverse=True)
        for token, freq in sorted_dict:
            if freq == 1:
                unk_morpheme_list.add(token)
            if token not in item_to_id:
                item_to_id[token] = len(item_to_id)
        return item_to_id, unk_morpheme_list, max_morph_per_word

    def build_oracle_vocab(self):
        max_morph_per_word = 0
        morpheme_dict = dict()

        morpheme_dict['<unk>'] = len(morpheme_dict)
        morpheme_dict['<PAD>'] = len(morpheme_dict)
        if self.eos != '':
            morpheme_dict[self.eos] = len(morpheme_dict)
        if self.sos != '':
            morpheme_dict[self.sos] = len(morpheme_dict)

        splitter = "+"
        for token in self.train_data:
            if token == self.eos or token == self.sos:
                continue
            morphemes = token.split(splitter)
            if splitter in token:
                # remove the word form
                morphemes = morphemes[1:]
            if len(morphemes) > max_morph_per_word:
                max_morph_per_word = len(morphemes)
            for morpheme in morphemes:
                if self.use_all_morphemes:
                    if morpheme not in morpheme_dict:
                        morpheme_dict[morpheme] = len(morpheme_dict)
                else:
                    if "lemma:" in morpheme or "pos:" in morpheme or "stem:" in morpheme:
                        if morpheme not in morpheme_dict:
                            morpheme_dict[morpheme] = len(morpheme_dict)
        return morpheme_dict, max_morph_per_word

    def replace_special_chars(self, word):
        """
        Replace special characters since we want to use them for
        the start and beginning of word symbols
        """
        word = re.sub("\^", "¬", word)
        word = re.sub("\$", "£", word)
        return word

    def read_dataset(self, filename):
        """
        Read data set from a file and put them into a list of tokens
        :param filename: file to read
        :return: data
        """
        data = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if self.lowercase or self.unit == "oracle":
                    line = line.lower()
                if self.sos != '':
                    data.append(self.sos)
                for word in line.split():
                    word = self.replace_special_chars(word)
                    _word = word
                    if self.unit == "oracle":
                        if "+" in word:
                            _word = word.split('+')[0].split(":")[1]
                    if self.unit == "morpheme":
                            _word = re.sub("@@", "", word)
                    if not self.is_hyperlink(_word.lower()) and len(_word) <= 100:
                        data.append(word)
                if self.eos != '':
                    data.append(self.eos)
        return data

    def read_words(self):
        """
        Read sequence of tokens from a given file
        This function is used to build dictionary
        """
        # If lowercase is True, it is already handled when we read the dataset.
        # If it is False, the unit must be other than word, so that we need to lowercase
        # the data since the word lookup table is for target words which are
        # always in lowercase.
        data = self.train_data
        if not self.lowercase or self.unit == "oracle":
            tmp_data = []
            for word in data:
                if self.unit == "oracle":
                    if '+' in word:
                        tags = word.split('+')
                        word_tag = tags[0].split(':')
                        word = word_tag[1]
                if self.unit == "morpheme":
                    word = re.sub("@@", "", word)
                word = word.lower()
                tmp_data.append(word)
            data = tmp_data
        return data

    def read_chars(self):
        """
        Read sequence of chars from a given file
        """
        char_data = []
        for word in self.train_data:
            if word == self.eos or word == self.sos:
                continue
            if self.unit == "oracle":
                if '+' in word:
                    tags = word.split('+')
                    word_tag = tags[0].split(':')
                    word = word_tag[1]
            if self.unit == "morpheme":
                word = re.sub("@@", "", word)
            char_data.extend([ch for ch in word])
        return char_data

    def get_max_word_length(self, word_dict):
        """
        Get maximum word length from the vocabulary
        :param word_dict: the vocabulary
        :return: maximum word length
        """
        max_len = 0
        max_word = ""
        for word in word_dict:
            word = "^" + word + "$"
            if len(word) > max_len:
                max_len = len(word)
                max_word = word
        print("Longest word: " + max_word + " " + str(max_len))
        return max_len

    def add_to_dict(self, _dict, start, end):
        """
        Add special symbols
        :param _dict: the dictionary
        :param start: start symbol
        :param end: end symbol
        :return: dictionary with counts
        """
        symbols = ['<unk>', start, end, '<PAD>']
        if self.eos != '':
            symbols.append(self.eos)
        if self.sos != '':
            symbols.append(self.sos)
        for s in symbols:
            _dict[s] = len(_dict)
        return _dict

    def data_to_word_ids(self, input_data, filter=False):
        """
        Given a list of words, convert each word into it's word id
        :param input_data: a list of words
        :return: a list of word ids
        """

        _buffer = list()
        for word in input_data:
            word = word.lower()
            if self.unit == "oracle":
                if "+" in word:
                    tokens = word.split('+')
                    word_tag = tokens[0].split(':')
                    word = word_tag[1]
            if self.unit == "morpheme":
                word = re.sub("@@", "", word)

            # flag to randomize token with frequency one
            flag = 1
            if word in self.unk_word_list:
                flag = random.randint(0, 1)

            if word in self.word_to_id and flag == 1:
                # if filter is True, reduce output vocabulary for softmax
                # (map words not in top self.out_vocab_size to UNK)
                if filter:
                    # index start from 0
                    if self.word_to_id[word] < self.out_vocab_size:
                        _buffer.append(self.word_to_id[word])
                    else:
                        _buffer.append(self.word_to_id['<unk>'])
                else:
                    _buffer.append(self.word_to_id[word])
            else:
                _buffer.append(self.word_to_id['<unk>'])
        return _buffer

    def create_binary_morpheme_vector(self, word):
        """
        Encode word into a binary vector of morphemes
        :param word: input word
        :return: a vector unit of the input word
        """
        dimension = len(self.morpheme_to_id)
        encoding = np.zeros(dimension)
        if word == self.eos or word == self.sos:
            encoding[self.morpheme_to_id[word]] = 1
        else:
            if self.unit == "morpheme":
                word = "^" + word + "$"
                morphemes = word.split("@@")
            else:  # oracle model
                morphemes = word.split("+")
                if "+" in word:
                    # remove the original word form
                    morphemes = morphemes[1:]

            for morpheme in morphemes:
                if self.unit == "morpheme":
                    if morpheme in self.morpheme_to_id:
                        encoding[self.morpheme_to_id[morpheme]] = 1
                    else:
                        for ch in morpheme:
                            flag = 1
                            if ch in self.unk_char_list:
                                flag = random.randint(0, 1)
                            if ch in self.morpheme_to_id and flag == 1:
                                encoding[self.morpheme_to_id[ch]] = 1
                            else:
                                encoding[self.morpheme_to_id['<unk>']] = 1
                else:  # oracle model
                    if morpheme in self.morpheme_to_id:
                        if self.use_all_morphemes:
                            encoding[self.morpheme_to_id[morpheme]] = 1
                        else:
                            if "lemma:" in morpheme or "pos:" in morpheme or "stem:" in morpheme:
                                encoding[self.morpheme_to_id[morpheme]] = 1
                    else:
                        encoding[self.morpheme_to_id['<unk>']] = 1
        return encoding

    def word_to_morphemes(self, word):
        """
        Encode word into a vector of its morpheme ids
        :param word: input word
        :return: a vector morphemic representation of the word
        """
        encoding = list()
        if word == self.eos or word == self.sos:
            encoding.append(self.morpheme_to_id[word])
        else:
            if self.unit == "morpheme":
                word = "^" + word + "$"
                morphemes = word.split("@@")
            else:  # oracle model
                morphemes = word.split("+")
                if "+" in word:
                    # remove the original word form
                    morphemes = morphemes[1:]
            for morpheme in morphemes:
                if self.unit == "morpheme":
                    if morpheme in self.morpheme_to_id:
                        encoding.append(self.morpheme_to_id[morpheme])
                    else:
                        for ch in morpheme:
                            flag = 1
                            if ch in self.unk_char_list:
                                flag = random.randint(0, 1)
                            if ch in self.morpheme_to_id and flag == 1:
                                encoding.append(self.morpheme_to_id[ch])
                            else:
                                encoding.append(self.morpheme_to_id['<unk>'])
                else:  # oracle model
                    if morpheme in self.morpheme_to_id:
                        if self.use_all_morphemes:
                            encoding.append(self.morpheme_to_id[morpheme])
                        else:
                            if "lemma:" in morpheme or "pos:" in morpheme or "stem:" in morpheme:
                                encoding.append(self.morpheme_to_id[morpheme])
                    else:
                        encoding.append(self.morpheme_to_id['<unk>'])
        return encoding

    def morpheme_encoding(self, data):
        """
        Given a list of words, convert each word into its morpheme vector
        :param data: a list of words
        :return: a list of morpheme vectors
        """
        _buffer = list()
        for word in data:
            if self.composition == "addition":
                _buffer.append(self.create_binary_morpheme_vector(word))
            elif self.composition == "bi-lstm":
                morphemes = self.word_to_morphemes(word)
                _buffer.append(self.padding(morphemes, self.max_morph_per_word,
                                            self.morpheme_to_id))
            else:
                sys.exit("Unknown composition")
        return _buffer

    def create_binary_ngram_vector(self, word, n):
        """
        Encode word into a binary vector of character ngrams
        :param word: input word
        :param n: the length of character to encode (n-gram)
        :return: a vector unit of the input word
        """
        dimension = len(self.ngram_to_id)
        encoding = np.zeros(dimension)
        if word == self.eos or word == self.sos:
            encoding[self.ngram_to_id[word]] = 1
        else:
            _word = '^' + word + '$'
            for i in range(len(_word) - n + 1):
                ngram = _word[i:i+n]
                if ngram in self.ngram_to_id:
                    encoding[self.ngram_to_id[ngram]] = 1
                else:
                    for ch in ngram:
                        flag = 1
                        if ch in self.unk_char_list:
                            flag = random.randint(0, 1)
                        if ch in self.ngram_to_id and flag == 1:
                            encoding[self.ngram_to_id[ch]] = 1
                        else:
                            encoding[self.ngram_to_id['<unk>']] = 1
        return encoding

    def word_to_ngrams(self, word):
        """
        Encode word into a vector of its ngram ids
        :param word: input word
        :return: a vector ngram representation of the word
        """
        encoding = list()
        n = self.n
        if word == self.eos or word == self.sos:
            encoding.append(self.ngram_to_id[word])
        else:
            _word = '^' + word + '$'
            for i in range(len(_word) - n + 1):
                ngram = _word[i:i + n]
                if ngram in self.ngram_to_id:
                    encoding.append(self.ngram_to_id[ngram])
                else:
                    for ch in ngram:
                        flag = 1
                        if ch in self.unk_char_list:
                            flag = random.randint(0, 1)
                        if ch in self.ngram_to_id and flag == 1:
                            encoding.append(self.ngram_to_id[ch])
                        else:
                            encoding.append(self.ngram_to_id['<unk>'])
        return encoding

    def ngram_encoding(self, data):
        """
        Given a list of words, convert each word into its character ngram unit
        :param input_data: a list of words
        :return: a list of character ngram unit of words
        """
        _buffer = list()
        for word in data:
            if self.composition == "addition":
                _buffer.append(self.create_binary_ngram_vector(word, self.n))
            elif self.composition == "bi-lstm":
                ngrams = self.word_to_ngrams(word)
                _buffer.append(self.padding(ngrams, self.max_ngram_per_word,
                                            self.ngram_to_id))
            else:
                sys.exit("Unknown composition")
        return _buffer

    def word_to_chars(self, word):
        """
        Break word into a sequence of characters
        :param word: a word
        :return: a list of character ids of the word
        """
        chars = list()
        if word == self.eos or word == self.sos:
            chars.append(self.char_to_id[word])
        else:
            word = "^" + word + "$"
            for ch in word:
                flag = 1
                if ch in self.unk_char_list:
                    flag = random.randint(0, 1)
                if ch in self.char_to_id and flag == 1:
                    chars.append(self.char_to_id[ch])
                else:
                    chars.append(self.char_to_id['<unk>'])
        return chars

    def char_encoding(self, data):
        """
        Given a list of words, convert each word into a list of characters
        :param data: a list of words
        :return: [a list of characters]
        """
        _buffer = list()
        for word in data:
            chars = self.word_to_chars(word)
            _buffer.append(self.padding(chars, self.max_word_len, self.char_to_id))
        return _buffer

    def encode_data(self, data):
        """
        Encode data according to the specified unit
        :param data: input data
        :return: encoded input data
        """
        if self.unit == "char":
            data = self.char_encoding(data)
        elif self.unit == "char-ngram":
            data = self.ngram_encoding(data)
        elif self.unit == "morpheme" or self.unit == "oracle":
            data = self.morpheme_encoding(data)
        else:
            data = self.data_to_word_ids(data, False)
        return data

    def data_iterator(self, raw_data, batch_size, num_steps):
        data_len = len(raw_data)
        batch_len = data_len // batch_size
        data = []
        for i in range(batch_size):
            x = raw_data[batch_len * i:batch_len * (i + 1)]
            data.append(x)

        epoch_size = (batch_len - 1) // num_steps

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            xs = list()
            ys = list()

            for j in range(batch_size):
                x = data[j][i * num_steps:(i + 1) * num_steps]
                y = data[j][i * num_steps + 1:(i + 1) * num_steps + 1]

                xs.append(self.encode_data(x))
                ys.append(self.data_to_word_ids(y, True))

            yield (xs, ys)

    def data_iterator_test(self, raw_data, batch_size, num_steps):
        data_len = len(raw_data)
        batch_len = data_len // batch_size
        data = []
        for i in range(batch_size):
            x = raw_data[batch_len * i:batch_len * (i + 1)]
            data.append(x)

        epoch_size = (batch_len - 1) // num_steps

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            xs = list()
            ys = list()

            for j in range(batch_size):
                x = data[j][i * num_steps:(i + 1) * num_steps]
                y = data[j][i * num_steps + 1:(i + 1) * num_steps + 1]

                xs.append(self.encode_data(x))
                ys.append(self.data_to_word_ids(y, True))

            yield (xs, ys)

