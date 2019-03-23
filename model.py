# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting#%%
import os
try:
    os.chdir(os.path.join(
        os.getcwd(), 'e_to_r_translation'))
    print(os.getcwd())
except:
    pass
# %%
import numpy as np
import tensorflow as tf
import time

from reader import Reader

in_file = "/home/raulslab/work/e_to_r_translation/data/europarl-v7.ro-en.en"
out_file = "/home/raulslab/work/e_to_r_translation/data/europarl-v7.ro-en.ro"

# %%


class Translator(object):
    def __init__(self, config):
        tf.reset_default_graph()
        self.ses = tf.Session()
        self.r = Reader(config.in_file, config.out_file,
                        config.vocab_size, config.batch_size)
        self.input_encoder, self.input_decoder, self.nr_steps_enc, self.nr_steps_dec = self.r.translator_batch_producer(
            self.ses, "Translator_Eng_Ro")

        self._batch_size = config.batch_size
        self._vocab_size = config.vocab_size
        self._keep_prob = config.keep_prob
        self._hidden_size = config.hidden_size
        self._layer_nr = config.layer_nr
        self._max_grad_norm = config.max_grad_norm
        self.init = tf.random_uniform_initializer(-0.1, 0.1)
        self._is_training = True
        self._epoch_size = config.epoch_size
        print("Out of init")

    def model(self, ):
        # Finally the fun part
        with tf.variable_scope("Translator", reuse=tf.AUTO_REUSE, initializer=self.init):
            inputs_enc = self.embed(self.input_encoder, "embedding_encoder")
            inputs_dec = self.embed(self.input_decoder, "embedding_decoder")

            encoder_state = self.build_encoder(inputs_enc, self._is_training)
            decoder_output = self.build_decoder(encoder_state, inputs_dec, self._is_training )                
            logits = self.fully_connected(decoder_output)

            logits = tf.reshape(logits, [self._batch_size, self.nr_steps_dec, self._vocab_size])

            loss = tf.contrib.seq2seq.sequence_loss(
                logits, self.input_decoder,
                tf.ones([self._batch_size, self.nr_steps_dec]),
                average_across_timesteps=False,
                average_across_batch=True
                )

            self._cost = tf.reduce_sum(loss)
            self._lr = tf.Variable(0.03, trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          self._max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step())

            self._new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")
            self._lr_update = tf.assign(self._lr, self._new_lr)
            
        return

    def embed(self, ids, name):
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                name=name, shape=[self._vocab_size, self._hidden_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(params=embedding, ids=ids)
            return inputs

    def make_cell(self, is_training):
        print("=========Create LSTM Cell")
        cell = tf.contrib.rnn.LSTMCell(num_units=self._hidden_size,
                                       use_peepholes=True)
        if is_training and self._keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell, output_keep_prob=self._keep_prob)
        return cell

    def build_encoder(self, inputs_enc, is_training):
        layers_enc = [self.make_cell(is_training)
                      for _ in range(self._layer_nr)]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(
            layers_enc, state_is_tuple=True)

        initial_zero_state = multi_layer_cell.zero_state(
            self._batch_size, tf.float32)
        _, states = tf.nn.dynamic_rnn(multi_layer_cell, inputs_enc, sequence_length=tf.fill([self._batch_size,],self.nr_steps_enc),
                                      initial_state=initial_zero_state, dtype=tf.float32)
        return states

    def build_decoder(self, encoder_state, input_decoder, is_training):
        layers_dec = [self.make_cell(is_training)
                      for _ in range(self._layer_nr)]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(
            layers_dec, state_is_tuple=True)

        outputs, _ = tf.nn.dynamic_rnn(multi_layer_cell, input_decoder, sequence_length=tf.fill([self._batch_size,], self.nr_steps_dec),
                                            initial_state=encoder_state, dtype=tf.float32)
        outputs = tf.reshape(outputs, [-1, self._hidden_size])
        return outputs

    def fully_connected(self, decoder_output):
        W = tf.get_variable("Weights_fc",dtype=tf.float32, shape=[self._hidden_size, self._vocab_size])
        b = tf.get_variable("Bias_fc", dtype=tf.float32, shape=[self._vocab_size])
        return tf.matmul(decoder_output, W) + b

    def train(self):
        self._is_training = True
        return {
            "cost": self._cost,
            "train_op": self._train_op
            #change_input_to_train_input
        }

    def valdiate(self):
        self._is_training = False
        return {
            "cost": self._cost
            #change_input_to_train_input
        }

    def test(self):
        self._is_training = False
        return {
            "cost": self._cost
            #change_input_to_train_input
        }

    def run_model(self, operation, verbose=False):
        start_time = time.time()
        costs = 0.0
        iters = 0
        for step in range(self._epoch_size):
            vals = self.ses.run(operation)
            costs += vals["cost"]
            iters += self.nr_steps_dec

            if verbose:
                print("Nr : %.3f perplexity: %.3f speed: %.0f wps" %
                    (step * 1.0 / self._epoch_size, np.exp(costs / iters),
                    (iters + self.nr_steps_enc) * self._batch_size /
                    (time.time() - start_time)))    

    def initialize_variables(self):
        self.ses.run(tf.global_variables_initializer())

    def change_config(self, new_config):
        return

    def tensordboard_write(self):
        writer = tf.summary.FileWriter('./graphs', self.ses.graph)

    def close_translator(self):
        self.r.free_threads()
        self.ses.close()

    def debug_print(self):
        print("ENCODER")
        print(self.ses.run(t.input_encoder[0]))
        print("DECODER")
        print(self.ses.run(t.input_decoder[0]))




class NormalConfig(object):
    in_file = in_file
    out_file = out_file
    layer_nr = 4
    batch_size = 100
    vocab_size = 100000
    keep_prob = 1.0
    hidden_size = 150
    max_grad_norm = 5
    epoch_size = 10


# class ValidationConfig(object):
#     in_file = in_file
#     out_file = out_file
#     layer_nr = 4
#     batch_size = 100
#     vocab_size = 100000
#     keep_prob = 1.0
#     hidden_size = 150
#     max_grad_norm = 5
#     epoch_size = 10

# class TestConfig(object):
#     in_file = in_file
#     out_file = out_file
#     layer_nr = 4
#     batch_size = 100
#     vocab_size = 100000
#     keep_prob = 1.0
#     hidden_size = 150
#     max_grad_norm = 5
#     epoch_size = 10

#%%
# Testing
t = Translator(NormalConfig())

t.model()
t.initialize_variables()
t.run_model(t.train(), verbose=True)

t.tensordboard_write()
t.close_translator()


#%%
