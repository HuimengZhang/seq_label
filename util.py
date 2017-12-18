# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import gzip
import logging
import sys

import numpy as np
import os
import os.path
import tensorflow as tf
from nltk import FreqDist

if (sys.version_info > (3, 0)):
    pass
else:  # Python 2.7 imports
    from io import open


def LoadEmbedding(embeddingsPath, commentSymbol=None):
    # Check that the embeddings file exists
    if not os.path.isfile(embeddingsPath):
        print("The embeddings file %s was not found" % embeddingsPath)
        exit()

    # :: Read in word embeddings ::
    logging.info("Read file: %s" % embeddingsPath)
    word2Idx = {}
    embeddings = []

    embeddingsIn = gzip.open(embeddingsPath, "rt") if embeddingsPath.endswith('.gz') else open(embeddingsPath)

    embeddingsDimension = None
    for line in embeddingsIn:
        if isinstance(line, unicode):
            split = line.rstrip().split(" ")
        else:
            split = line.rstrip().decode("utf8").split(" ")

        if len(split) > 2:
            word = split[0]
            if embeddingsDimension == None:
                embeddingsDimension = len(split) - 1

            if (len(
                    split) - 1) != embeddingsDimension:  # Assure that all lines in the embeddings file are of the same length
                print("ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
                continue

            if len(word2Idx) == 0:  # Add padding+unknown
                word2Idx["PADDING_TOKEN"] = len(word2Idx)
                vector = np.zeros(embeddingsDimension)
                embeddings.append(vector)
                word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
                vector = np.random.uniform(-0.25, 0.25, embeddingsDimension)  # Alternativ -sqrt(3/dim) ... sqrt(3/dim)
                embeddings.append(vector)

            vector = np.array([float(num) for num in split[1:]])

            if word not in word2Idx:
                embeddings.append(vector)
                word2Idx[word] = len(word2Idx)

    embeddings = np.array(embeddings)
    return embeddings, word2Idx


def readCoNLLTrain(inputPath, cols, word2idx, labelkey):
    """
    Reads in a CoNLL file
    """
    logging.info("Read file: %s" % inputPath)
    sentences = []
    sentenceTemplate = {name: [] for name in cols.values() + ['raw_tokens', 'raw_labels']}
    sentence = {name: [] for name in sentenceTemplate.keys()}
    label2idx = {"O": 0}
    newData = False
    numTokens = 0
    numUnknownTokens = 0
    missingTokens = FreqDist()

    for line in open(inputPath):
        line = line.strip()
        if len(line) == 0:
            if newData:
                sentences.append(sentence)
                sentence = {name: [] for name in sentenceTemplate.keys()}
                newData = False
            continue

        splits = line.split()
        for colIdx, colName in cols.items():
            val = splits[colIdx]
            if colName == 'tokens':
                numTokens += 1
                idx = word2idx['UNKNOWN_TOKEN']

                if word2idx.has_key(val):
                    idx = word2idx[val]
                else:
                    numUnknownTokens += 1
                    missingTokens[val] += 1
                sentence['raw_tokens'].append(val)
                sentence[colName].append(idx)
            elif colName == labelkey:
                if not label2idx.has_key(val):
                    label2idx[val] = len(label2idx)
                idx = label2idx[val]
                sentence[colName].append(idx)
                sentence['raw_labels'].append(val)
        newData = True

    if numTokens > 0:
        logging.info("Unknown-Tokens: %.2f%%" % (numUnknownTokens / float(numTokens) * 100))

    if newData:
        sentences.append(sentence)

    return sentences, label2idx


def readCoNLL(inputPath, cols, word2idx, labelkey, label2idx):
    """
    Reads in a CoNLL file
    """
    logging.info("Read file: %s" % inputPath)
    sentences = []
    sentenceTemplate = {name: [] for name in cols.values() + ['raw_tokens', 'raw_labels']}
    sentence = {name: [] for name in sentenceTemplate.keys()}
    newData = False
    numTokens = 0
    numUnknownTokens = 0
    missingTokens = FreqDist()

    for line in open(inputPath):
        line = line.strip()
        if len(line) == 0:
            if newData:
                sentences.append(sentence)
                sentence = {name: [] for name in sentenceTemplate.keys()}
                newData = False
            continue

        splits = line.split()
        for colIdx, colName in cols.items():
            val = splits[colIdx]
            if colName == 'tokens':
                numTokens += 1
                idx = word2idx['UNKNOWN_TOKEN']

                if val in word2idx.keys():
                    idx = word2idx[val]
                else:
                    numUnknownTokens += 1
                    missingTokens[val] += 1
                sentence['raw_tokens'].append(val)
                sentence[colName].append(idx)
            elif colName == labelkey:
                idx = label2idx[val]
                sentence[colName].append(idx)
                sentence['raw_labels'].append(val)
        newData = True

    if numTokens > 0:
        logging.info("Unknown-Tokens: %.2f%%" % (numUnknownTokens / float(numTokens) * 100))

    if newData:
        sentences.append(sentence)

    return sentences


def prepare(dataset, labelKey, seq_max_len, is_padding=True):
    X = []
    y = []
    tmp_x = []
    tmp_y = []

    for idx in range(len(dataset)):
        c = dataset[idx]['tokens']
        # print("c", c, "l", l)
        l = dataset[idx][labelKey]
        # empty line
        if is_padding:
            tmp_x.append(padding(c, seq_max_len))
            tmp_y.append(padding(l, seq_max_len))
        else:
            tmp_x.append(c)
            tmp_y.append(l)

    # print(X)
    return tmp_x, tmp_y


# use "0" to padding the sentence
def padding(Sequence, seq_max_len):
    seq_out = []
    if len(Sequence) < seq_max_len:
        for i in range(len(Sequence)):
            seq_out.append(Sequence[i])
        for i in range(len(Sequence), seq_max_len):
            seq_out.append(0)
    elif len(Sequence) >= seq_max_len:
        for i in range(seq_max_len):
            seq_out.append(Sequence[i])
    return seq_out


def create_model(session, Model, ckpt_file, labelKey, label2Idx, word2Idx, num_steps, num_epochs, embedding_matrix,
                 logger):
    # create model, reuse parameters if exists
    model = Model(labelKey=labelKey, label2Idx=label2Idx, word2Idx=word2Idx, num_steps=num_steps, num_epochs=num_epochs,
                  embedding_matrix=embedding_matrix, is_training=True)

    ckpt = tf.train.get_checkpoint_state(ckpt_file)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    return model


def nextBatch(X, y, start_index, batch_size):
    last_index = start_index + batch_size
    X_batch = list(X[start_index:min(last_index, len(X))])
    y_batch = list(y[start_index:min(last_index, len(X))])
    if last_index > len(X):
        left_size = last_index - (len(X))
        for i in range(left_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
            y_batch.append(y[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch


def nextRandomBatch(X, y, batch_size):
    X_batch = []
    y_batch = []
    for i in range(batch_size):
        index = np.random.randint(len(X))
        X_batch.append(X[index])
        y_batch.append(y[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch


def save_model(sess, model, path, model_saved_name, logger):
    checkpoint_path = os.path.join(path, model_saved_name)
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")


def cal_recall(guessed, correct, idx2Label):
    assert (len(guessed) == len(correct))
    label_pred = [idx2Label[element] for element in guessed]
    label_correct = [idx2Label[element] for element in correct]
    idx = 0
    correctCount = 0
    count = 0
    while idx < len(label_pred):
        if label_pred[idx][0] == 'B':  # A new chunk starts
            count += 1

            if label_pred[idx] == label_correct[idx]:
                idx += 1
                correctlyFound = True

                while idx < len(label_pred) and label_pred[idx][0] == 'I':  # Scan until it no longer starts with I
                    if label_pred[idx] != label_correct[idx]:
                        correctlyFound = False

                    idx += 1

                if idx < len(label_pred):
                    if label_correct[idx][0] == 'I':  # The chunk in correct was longer
                        correctlyFound = False

                if correctlyFound:
                    correctCount += 1
            else:
                idx += 1
        else:
            idx += 1

    return count, correctCount


def getTransition(y_train_batch, num_class):
    transition_batch = []
    for m in range(len(y_train_batch)):
        y = [num_class] + list(y_train_batch[m]) + [0]
        for t in range(len(y)):
            if t + 1 == len(y):
                continue
            i = y[t]
            j = y[t + 1]
            if i == 0:
                break
            transition_batch.append(i * (num_class + 1) + j)
    transition_batch = np.array(transition_batch)
    return transition_batch
