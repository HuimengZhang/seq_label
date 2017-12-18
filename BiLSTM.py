import math
import numpy as np
import tensorflow as tf
import logging
import util


class BILSTM_CRF(object):
    def __init__(self, labelKey, label2Idx, word2Idx, embedding_matrix, num_steps=200, num_epochs=100, weight=False, is_training=True, is_crf=True):
        # Parameter
        self.max_f1 = 0
        self.learning_rate_base = 0.001
        self.learning_rate_decay = 0.99
        self.global_step = tf.Variable(0, trainable=False)
        self.regularizer = None
        if is_training:
            self.regularizer = tf.contrib.layers.l2_regularizer(0.001)
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_base, self.global_step, 1000, self.learning_rate_decay, staircase=True)
        self.l2_reg = 0.0001
        self.dropout_rate = 0.5
        self.batch_size = 50
        self.num_layers = 1
        self.emb_dim = 200
        self.hidden_dim = 100
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.char2id = word2Idx
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.labelKey = labelKey
        self.label2id = label2Idx
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_chars = len(self.id2char)
        self.num_classes = len(self.label2id)

        # placeholder of x, y and weight
        self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets_transition = tf.placeholder(tf.int32, [None])
        self.targets_weight = tf.placeholder(tf.float32, [None, self.num_steps])

        # char embedding
        with tf.variable_scope("embeddings"):
            self.embedding = embedding_matrix
            if is_training:
                self.embedding = tf.nn.dropout(self.embedding, self.dropout_rate)

            self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)  # shape=(?, 200, 200)
            # self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])                    #shape=(200, ?, 200)
            # self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim])          #shape=(?, 200)
            self.model_inputs = tf.cast(self.inputs_emb, tf.float32)
            # self.inputs_emb = tf.split(self.inputs_emb, self.num_steps, 0)                #split into 200 tensor each shape = (?, 200)
        # lstm cell
        with tf.variable_scope("BILSTM"):
            lstm_cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_dim)
            lstm_cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_dim)

            # dropout
            if is_training:
                lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
                lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))

            lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_fw] * self.num_layers)
            lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_bw] * self.num_layers)

            # get the length of each sample
            self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
            self.length = tf.cast(self.length, tf.int32)

            # forward and backward
            self.outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                # input demension =[batch_size , num_timesteps , num_features.]
                lstm_cell_fw,
                lstm_cell_bw,
                self.model_inputs,
                dtype=tf.float32,
                sequence_length=self.length
            )

        # softmax
        with tf.variable_scope("softmax"):
            self.outputs = tf.concat([self.outputs[0], self.outputs[1]], 2)
            self.outputs = tf.reshape(self.outputs, [-1, self.hidden_dim * 2])

            self.softmax_w = self.get_weight_variable([self.hidden_dim * 2, self.num_classes], self.regularizer)
            tf.get_variable(name="softmax_w", shape=[self.hidden_dim * 2, self.num_classes],
                                             initializer=self.initializer)
            self.softmax_b = tf.Variable(tf.zeros([self.num_classes], name="softmax_b"))
            self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
            self.logits = tf.reshape(self.logits, [-1, self.num_steps, self.num_classes], name="unary_scores")

        with tf.variable_scope("crf"):
            if not is_crf:
                pass
            else:
                log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(inputs=self.logits,
                                                                                           tag_indices=self.targets,
                                                                                           sequence_lengths=self.length)
                self.loss = tf.reduce_mean(-log_likelihood)
                self.loss = self.loss + tf.add_n(tf.get_collection('losses'))


                # summary
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)
        # grads_vars = self.train_op.compute_gradients(self.loss)
        # capped_grads_vars = [[tf.clip_by_value(g, -5, 5), v] for g, v in grads_vars]
        # self.train_op = self.train_op.apply_gradients(capped_grads_vars, self.global_step)

        self.train_summary = tf.summary.scalar("loss", self.loss)
        self.val_summary = tf.summary.scalar("loss", self.loss)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def get_weight_variable(self, shape, regularizer):
        weights = tf.get_variable("weights", shape, initializer=self.initializer)
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weights))
        return weights

    def viterbi_decode(self, pred, y_gold, trans_matrix):
        """
        Given predicted unary_scores, using viterbi_decode find the best tags
        sequence.
        shape of y_gold = (128,200) = (batch_size, num_step)
        shape of pred = (128,200,3) = (batch_size, num_step, num_class)
        """

        labels = []
        for i in xrange(self.batch_size):
            p_len = y_gold[i]
            unary_scores = pred[i][:len(p_len)]
            tags_seq, _ = tf.contrib.crf.viterbi_decode(unary_scores, trans_matrix)
            labels.append(np.array(tags_seq))
        return labels

    def run_step(self, sess, is_train, X, y):

        feed_dict = {
            self.inputs: X,
            self.targets: y
        }
        if is_train:
            _, unary_scores, loss, transition_params, length, summary, lr = \
                sess.run([
                    self.train_op,
                    self.logits,
                    self.loss,
                    self.transition_params,
                    self.global_step,
                    self.train_summary,
                    self.learning_rate

                ], feed_dict)
        else:
            _, unary_scores, loss, transition_params, length, summary, lr = \
                sess.run([
                    self.train_op,
                    self.logits,
                    self.loss,
                    self.transition_params,
                    self.length,
                    self.val_summary,
                    self.learning_rate
                ], feed_dict)
        return length, lr, unary_scores, loss, transition_params, summary

    def train(self, sess, save_file, X_train, y_train, X_val, y_val, save_model_name, logger):

        merged = tf.summary.merge_all()
        summary_writer_train = tf.summary.FileWriter('loss_log/train_loss', sess.graph)
        summary_writer_val = tf.summary.FileWriter('loss_log/val_loss', sess.graph)

        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))
        cnt = 0
        Pre_Count = 0
        Pre_Correct = 0
        Rec_Count = 0
        Rec_Correct = 0
        for epoch in range(self.num_epochs):
            print ("current epoch: %d" % (epoch))
            for iteration in range(num_iterations):
                # train
                num_samples = len(X_train)
                indexs = np.arange(num_samples)
                np.random.shuffle(indexs)
                train_x = [X_train[idx] for idx in indexs]
                train_y = [y_train[idx] for idx in indexs]

                X_train_batch, y_train_batch = util.nextBatch(train_x, train_y,
                                                              start_index=iteration * self.batch_size,
                                                              batch_size=self.batch_size)
                global_step, lr, unary_scores, loss_train, transition_params, train_summary = self.run_step(
                    sess, True,
                    X_train_batch,
                    y_train_batch)
                predicts_train = self.viterbi_decode(unary_scores, y_train_batch, transition_params)
                cnt += 1
                Correct_Tag, Total_Tag, Correct_Token, Total_Token = self.evaluate(y_train_batch, predicts_train)
                Pre_Count += Total_Tag
                Pre_Correct += Correct_Tag
                Rec_Count += Total_Token
                Rec_Correct += Correct_Token

                if iteration % 100 == 0:
                    print ("iteration: %5d, loss_train: %5d, learning rate: %.5f" % (iteration, loss_train, lr))
                summary_writer_train.add_summary(train_summary, cnt)
            if Pre_Count > 0:
                precision_train = Pre_Correct / float(Pre_Count)
            else:
                precision_train = 0
            if Rec_Count > 0:
                recall_train = Rec_Correct / float(Rec_Count)
            else:
                recall_train = 0
            if (precision_train + recall_train) > 0:
                f1_train = 2 * precision_train * recall_train / float(precision_train + recall_train)
            else:
                f1_train = 0
            print ('train precision: %.5f, train recall: %.5f, train f1: %.5f, learning_rate: %.5f' % (
            precision_train, recall_train,
            f1_train, lr))

            # validation
            val_num_iterations = int(math.ceil(1.0 * len(X_val) / self.batch_size))
            cnt = 0
            Pre_Count = 0
            Pre_Correct = 0
            Rec_Count = 0
            Rec_Correct = 0
            for iteration_v in range(val_num_iterations):
                num_samples = len(X_val)
                indexs = np.arange(num_samples)
                np.random.shuffle(indexs)
                val_x = [X_val[idx] for idx in indexs]
                val_y = [y_val[idx] for idx in indexs]
                X_val_batch, y_val_batch = util.nextBatch(val_x, val_y,
                                                          start_index=iteration_v * self.batch_size,
                                                          batch_size=self.batch_size)
                length, lr, v_unary_scores, loss_val, v_transition_params, val_summary = self.run_step(
                    sess, False,
                    X_val_batch,
                    y_val_batch)
                predicts_val = self.viterbi_decode(v_unary_scores, y_val_batch, v_transition_params)
                Correct_Tag, Total_Tag, Correct_Token, Total_Token = self.evaluate(y_val_batch, predicts_val)
                Pre_Count += Total_Tag
                Pre_Correct += Correct_Tag
                Rec_Count += Total_Token
                Rec_Correct += Correct_Token

                summary_writer_val.add_summary(val_summary, cnt)
            if Pre_Count > 0:
                precision_val = Pre_Correct / float(Pre_Count)
            else:
                precision_val = 0
            if Rec_Count > 0:
                recall_val = Rec_Correct / float(Rec_Count)
            else:
                recall_val = 0
            if (precision_val + recall_val) > 0:
                f1_val = 2 * precision_val * recall_val / float(precision_val + recall_val)
            else:
                f1_val = 0

            print(" valid precision: %.5f, valid recall: %.5f, valid f1: %.5f" % (precision_val, recall_val, f1_val))

            if f1_val > self.max_f1:
                self.max_f1 = f1_val
                util.save_model(sess, self, save_file, save_model_name, logger)
                print("saved the best model with f1: %.5f" % (self.max_f1))


    def evaluate(self, Input_label, Predict_label):
        Correct_Tag = 0
        Total_Tag = 0
        Total_Token = 0
        Correct_Token = 0
        y = []
        y_perdict = []
        for sentenceIdx in range(len(Input_label)):
            count = 0
            for idx in range(len(Input_label[sentenceIdx])):
                if Input_label[sentenceIdx][idx] == 0:
                    break
                else:
                    count += 1
            y.append(Input_label[sentenceIdx][0:idx])
            y_perdict.append(Predict_label[sentenceIdx][0:idx])
        for idx in range(len(y)):
            count_Token, count_Correct_Token = util.cal_recall(y[idx], y_perdict[idx], self.id2label)
            Total_Token += count_Token
            Correct_Token += count_Correct_Token
            count_Tag, count_Correct_Tag = util.cal_recall(y_perdict[idx], y[idx], self.id2label)
            Total_Tag += count_Tag
            Correct_Tag += count_Correct_Tag

        return Correct_Tag, Total_Tag, Correct_Token, Total_Token
