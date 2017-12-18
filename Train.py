from __future__ import print_function
import os
import logging
import sys
import time
import tensorflow as tf
from BiLSTM import BILSTM_CRF
import util

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

data_path = "ws_pku_BI"
save_path = "saved_model"
num_epochs = 200
embeddingsPath = "../newsblogbbs2.vec"
num_steps = 100  # it must consist with the test
dataColumns = {0: 'tokens', 1: 'WS_BI'}  # Tab separated columns, column 1 contains the token, 3 the universal POS tag
labelKey = 'WS_BI'

start_time = time.time()
logger.info("preparing train and validation data")
save_model_name = data_path+'_'+labelKey+'2'

######################################################
#
# Data preprocessing
#
######################################################


# :: Train / Dev / Test-Files ::

# Parameters of the network

trainData = '../data/%s/train.txt' % data_path
# devData = 'data/%s/dev.txt' % data_path
testData = '../data/%s/test.txt' % data_path
datasetFiles = [trainData, testData]
# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
embeddings, word2Idx = util.LoadEmbedding(embeddingsPath)

trainSentences, label2Idx = util.readCoNLLTrain(datasetFiles[0], dataColumns, word2Idx, labelKey)
if len(datasetFiles) == 3:
    devSentences = util.readCoNLL(datasetFiles[1], dataColumns, word2Idx, labelKey, label2Idx)
    testSentences = util.readCoNLL(datasetFiles[2], dataColumns, word2Idx, labelKey, label2Idx)
elif len(datasetFiles) == 2:
    devSentences = util.readCoNLL(datasetFiles[1], dataColumns, word2Idx, labelKey, label2Idx)

print("embeddings", embeddings.shape[0], embeddings.shape[1])  # 11210,200 (the numbr of words, embedding size)

print("Train Sentences:", len(trainSentences))
print("Test Sentences:", len(devSentences))


X_train, y_train = util.prepare(dataset=trainSentences, labelKey=labelKey, seq_max_len=num_steps)
X_dev, y_dev = util.prepare(dataset=devSentences, labelKey=labelKey, seq_max_len=num_steps)

num_chars = len(word2Idx.keys())
num_classes = len(label2Idx.keys())

print(num_chars, num_classes)

######################################################
#
# Model Running
#
######################################################

logger.info("building model")
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    initializer = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        model = util.create_model(
            session=sess,
            Model=BILSTM_CRF,
            ckpt_file=save_path,
            labelKey=labelKey,
            label2Idx=label2Idx,
            word2Idx=word2Idx,
            num_steps=num_steps,
            num_epochs=num_epochs,
            embedding_matrix=embeddings,
            logger=logger)
        logger.info("training model")
        logger.info("start training")
        model.train(sess, save_path, X_train, y_train, X_dev, y_dev, save_model_name, logger)

        print("final best f1 is: %f" % (model.max_f1))

        end_time = time.time()
        print("time used %f(hour)" % ((end_time - start_time) / 3600))

