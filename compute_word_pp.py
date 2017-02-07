#!/usr/bin/python
# Author: Clara Vania

import numpy as np
import tensorflow as tf
import argparse
import time
import math
import os, sys
import pickle

from collections import defaultdict
from utils import TextLoader
from biLSTM import BiLSTMModel
from add import AdditiveModel
from word import WordModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='../../data/multi/id/test.txt',
                        help="test file")
    parser.add_argument('--wordlist_file', type=str, default='../../data/multi/id/analysis/redup_words.txt',
                        help="word list file")
    parser.add_argument('--freq_type', type=str, default="freq",
                        help="frequency type: rare, freq, all")
    parser.add_argument('--save_dir', type=str, default='wiki_models/id/bpe.lstm',
                        help='directory of the checkpointed models')
    args = parser.parse_args()
    test(args)


def run_epoch(session, m, data, data_loader, eval_op, word_list):
    costs = 0.0
    iters = 0
    pp = 0.0
    state = session.run(m.initial_lm_state)

    if data_loader.composition == "bi-lstm":
        session.run(m.initial_fw_state)
        session.run(m.initial_bw_state)

    prev = False
    counter = 0
    max_counter = 1
    for step, (x, y, tokens) in enumerate(data_loader.data_iterator_test(data, m.batch_size, m.num_steps)):
        vector, per_word_pp, cost, state, _ = session.run([m.input_vectors, m.per_word_pp, m.cost,
                                                           m.final_state, eval_op],
                                                         {m.input_data: x,
                                                          m.targets: y,
                                                          m.initial_lm_state: state})

        token = tokens[0][0]
        word_pp = per_word_pp[0]

        # costs += word_pp
        # iters += 1

        if prev:
            costs += word_pp
            pp += np.exp(word_pp)
            # print(token, word_pp)
            iters += 1

        if token in word_list:
            prev = True
            counter = 1
        else:
            if counter >= max_counter:
                prev = False
                counter = 0
            else:
                counter += 1

    print("******************")
    print(pp)
    print(costs)
    print(iters)
    print(pp / iters)
    print(costs/iters)
    return np.exp(costs / iters)


def test(test_args):

    with open(os.path.join(test_args.save_dir, 'config.pkl'), 'rb') as f:
        args = pickle.load(f)

    args.save_dir = test_args.save_dir
    data_loader = TextLoader(args, train=False)
    test_data = data_loader.read_dataset(test_args.test_file)

    print(args.save_dir)
    print("Unit: " + args.unit)
    print("Composition: " + args.composition)
    print("Freq: " + test_args.freq_type)

    args.word_vocab_size = data_loader.word_vocab_size
    if args.unit != "word":
        args.subword_vocab_size = data_loader.subword_vocab_size

    # Statistics of words
    print("Word vocab size: " + str(data_loader.word_vocab_size))

    # Statistics of sub units
    if args.unit != "word":
        print("Subword vocab size: " + str(data_loader.subword_vocab_size))
        if args.composition == "bi-lstm":
            if args.unit == "char":
                args.bilstm_num_steps = data_loader.max_word_len
                print("Max word length:", data_loader.max_word_len)
            elif args.unit == "char-ngram":
                args.bilstm_num_steps = data_loader.max_ngram_per_word
                print("Max ngrams per word:", data_loader.max_ngram_per_word)
            elif args.unit == "morpheme" or args.unit == "oracle":
                args.bilstm_num_steps = data_loader.max_morph_per_word
                print("Max morphemes per word", data_loader.max_morph_per_word)

    if args.unit == "word":
        lm_model = WordModel
    elif args.composition == "addition":
        lm_model = AdditiveModel
    elif args.composition == "bi-lstm":
        lm_model = BiLSTMModel
    else:
        sys.exit("Unknown unit or composition.")

    word_list = {}
    with open(test_args.wordlist_file) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            word, freq = line.strip().split()
            freq = int(freq)
            if test_args.freq_type == "rare":
                if freq <= 10:
                    word_list[word] = freq
            elif test_args.freq_type == "freq":
                if freq > 10:
                    word_list[word] = freq
            else:  # use all words
                word_list[word] = freq

    print("Begin testing...")
    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("model"):
            mtest = lm_model(args, is_training=False, is_testing=True)

        # save only the last model
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        # tf.initialize_all_variables().run()
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        test_perplexity = run_epoch(sess, mtest, test_data, data_loader, tf.no_op(), word_list)
        print("Test perplexity: %.3f" % test_perplexity)
        print("\n")


if __name__ == '__main__':
    main()
