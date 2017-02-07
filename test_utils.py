#!/usr/bin/python

import argparse
import os
from utils import TextLoader
import operator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='../../data/multi/cs/small/train.txt',
                        help="training data")
    parser.add_argument('--dev_file', type=str, default='../../data/multi/cs/small/dev.txt',
                        help="development data")
    parser.add_argument('--output_vocab_file', type=str, default='',
                        help="wiki_models/cs/word/word_vocab.pkl")
    parser.add_argument('--output', '-o', type=str, default='train.log',
                        help='output file')
    parser.add_argument('--save_dir', type=str, default='model',
                        help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=200,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--num_highway', type=int, default=2,
                        help='number of highway layers (for CNN model)')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--unit', type=str, default='word',
                        help='char, char-ngram, morpheme, word, or oracle')
    parser.add_argument('--composition', type=str, default='none',
                        help='none(word), addition, or bi-lstm')
    parser.add_argument('--lowercase', dest='lowercase', action='store_true',
                        help='lowercase data', default=False)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='RNN sequence length')
    parser.add_argument('--out_vocab_size', type=int, default=5000,
                        help='size of output vocabulary')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--patience', type=int, default=3,
                        help='the number of iterations allowed before decaying the '
                             'learning rate if there is no improvement on dev set')
    parser.add_argument('--validation_interval', type=int, default=1,
                        help='validation interval')
    parser.add_argument('--init_scale', type=float, default=0.1,
                        help='initial weight scale')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='maximum permissible norm of the gradient')
    parser.add_argument('--learning_rate', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='the decay of the learning rate')
    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='the probability of keeping weights in the dropout layer')
    parser.add_argument('--gpu', type=int, default=0,
                        help='the gpu id, if have more than one gpu')
    parser.add_argument('--optimization', type=str, default='sgd',
                        help='sgd, momentum, or adagrad')
    parser.add_argument('--train', type=str, default='softmax',
                        help='sgd, momentum, or adagrad')
    parser.add_argument('--SOS', type=str, default='true',
                        help='start of sentence symbol')
    parser.add_argument('--EOS', type=str, default='true',
                        help='end of sentence symbol')
    parser.add_argument('--ngram', type=int, default=3,
                        help='length of character ngram (for char-ngram model only)')
    parser.add_argument('--char_dim', type=int, default=200,
                        help='dimension of char embedding (for C2W model only)')
    parser.add_argument('--morph_dim', type=int, default=200,
                        help='dimension of morpheme embedding (for M2W model only)')
    parser.add_argument('--word_dim', type=int, default=200,
                        help='dimension of word embedding (for C2W model only)')
    parser.add_argument('--cont', type=str, default='false',
                        help='continue training')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for random initialization')
    args = parser.parse_args()
    train(args)


def train(args):
    save_dir = args.save_dir
    try:
        os.stat(save_dir)
    except:
        os.mkdir(save_dir)

    args.eos = ''
    args.sos = ''
    if args.EOS == "true":
        args.eos = '</s>'
        args.out_vocab_size += 1
    if args.SOS == "true":
        args.sos = '<s>'
        args.out_vocab_size += 1

    data_loader = TextLoader(args)
    data = data_loader.dev_data

    print(data[0:50])

    sorted_dict = sorted(data_loader.word_to_id.items(), key=operator.itemgetter(1))
    fout = open("vocab.txt", "w")
    for k, v in sorted_dict:
        fout.write(str(k) + " " + str(v) + "\n")
    fout.close()

    sorted_dict = sorted(data_loader.char_to_id.items(), key=operator.itemgetter(1))
    fout = open("char_vocab.txt", "w")
    for k, v in sorted_dict:
        fout.write(str(k) + " " + str(v) + "\n")
    fout.close()

    if args.unit == "morpheme" or args.unit == "oracle":
        sorted_dict = sorted(data_loader.morpheme_to_id.items(), key=operator.itemgetter(1))
        fout = open("morph_vocab.txt", "w")
        for k, v in sorted_dict:
            fout.write(str(k) + " " + str(v) + "\n")
        fout.close()
    elif args.unit == "char-ngram":
        sorted_dict = sorted(data_loader.ngram_to_id.items(), key=operator.itemgetter(1))
        fout = open("morph_vocab.txt", "w")
        for k, v in sorted_dict:
            fout.write(str(k) + " " + str(v) + "\n")
        fout.close()

    for step, (x, y) in enumerate(data_loader.data_iterator(data, 1, 5)):
        if step < 10:
            print(x)
            print(y)
        else:
            break
        print("***********************************")

if __name__ == '__main__':
    main()