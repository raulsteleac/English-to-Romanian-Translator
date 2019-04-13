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
        self.in_enc, self.in_dec, self.nr_steps_in, self.nr_steps_out, self.test_enc, self.test_nr_steps_i, self.test_dec, self.test_nr_steps_o, self._starting_string, self._terminating_string = self.r.translator_batch_producer(self.ses, "Translator_Eng_Ro")

        #Initialize the necessary parameters for training 
        self.input_encoder = self.in_enc
        self.input_decoder = self.in_dec
        self.nr_steps_enc = self.nr_steps_in
        self.nr_steps_dec = self.nr_steps_out
        self._batch_nr = self.r._batch_nr
        self._is_training = True

        #Initialize the model configuration
        self._batch_size = config.batch_size
        self._epoch_size = config.epoch_size
        self._hidden_size = config.hidden_size
        self._keep_prob = config.keep_prob
        self._layer_nr = config.layer_nr
        self._max_grad_norm = config.max_grad_norm
        self._training_slice = config.training_slice
        self._test_slice = config.test_slice
        self._validation_slice = config.validation_slice
        self._vocab_size = config.vocab_size

        self.init = tf.random_uniform_initializer(-0.1, 0.1)
        print("Out of init")

    def model(self, ):
        # Finally the fun part
        with tf.variable_scope("Translator", reuse=tf.AUTO_REUSE, initializer=self.init):
            self.encoder_embeddings = tf.get_variable(name="embedding_encoder", 
                                                shape=[self._vocab_size, self._hidden_size],
                                                dtype=tf.float32)
            inputs_enc = self.embed(self.input_encoder, self.encoder_embeddings)
            encoder_state = self.build_encoder(inputs_enc, self._is_training)

            self.decoder_embeddings = tf.get_variable(name="embedding_decoder", shape=[self._vocab_size, self._hidden_size], dtype=tf.float32)

            dec_cell = self.get_decoder_cell(self._is_training)
            if self._is_training:
                #Decpder for training
                inputs_dec = self.embed(self.input_decoder, self.decoder_embeddings)
                decoder_output = self.build_decoder_train(dec_cell, encoder_state, inputs_dec)
            else:
                #Decoder for testing
                decoder_output = self.build_decoder_infer(dec_cell, encoder_state)

            with tf.name_scope("Fully_Connected_Layer"):
                logits = self.fully_connected(decoder_output)
                logits = tf.reshape(logits, [self._batch_size, self.nr_steps_dec, self._vocab_size])

            loss = tf.contrib.seq2seq.sequence_loss(logits, self.input_decoder,
                                                    tf.ones([self._batch_size, self.nr_steps_dec]),
                                                    average_across_timesteps=False,
                                                    average_across_batch=True)
            self.output = logits
            self._cost = tf.reduce_sum(loss)
            self._lr = tf.Variable(0.03, trainable=False)

            tf.summary.scalar('Cost', self._cost)

            if not self._is_training:
                return

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                                           self._max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.train.get_or_create_global_step())
            self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
            self._lr_update = tf.assign(self._lr, self._new_lr)            
        return

    def embed(self, ids, embedding):
        with tf.device("/cpu:0"):
            inputs = tf.nn.embedding_lookup(params=embedding, ids=ids)
            return inputs

    def make_cell(self, is_training):
        print("=========Create LSTM Cell")
        cell = tf.contrib.rnn.LSTMCell(num_units=self._hidden_size, use_peepholes=True)
        if is_training and self._keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob)
        return cell
    
    def get_decoder_cell(self, is_training):
        layers_dec = [self.make_cell(is_training) for _ in range(self._layer_nr)]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers_dec, state_is_tuple=True)    
        return multi_layer_cell
    
    def get_encoder_cell(self, is_training):
        layers_enc = [self.make_cell(is_training) for _ in range(self._layer_nr)]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers_enc, state_is_tuple=True)
        return multi_layer_cell

    def build_encoder(self, inputs_enc, is_training):
        multi_layer_cell = self.get_encoder_cell(is_training)
        initial_zero_state = multi_layer_cell.zero_state(self._batch_size, tf.float32)
        _, states = tf.nn.dynamic_rnn(multi_layer_cell,
                                    inputs_enc, sequence_length=tf.fill([self._batch_size,],
                                    self.nr_steps_enc),
                                    initial_state=initial_zero_state,
                                    dtype=tf.float32)
        return states

    def build_decoder_train(self, dec_cell, encoder_state, input_decoder):
        sampling_probability = tf.Variable(0.8, dtype=tf.float32)
        helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(input_decoder, 
                                                                tf.fill([self._batch_size], self.nr_steps_dec),
                                                                sampling_probability=sampling_probability)
        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                        impute_finished=True,
                                                        maximum_iterations=self.nr_steps_dec)
        outputs = tf.reshape(outputs.rnn_output, [-1, self._hidden_size])
        return outputs

    def build_decoder_infer(self, dec_cell, encoder_state):
        with tf.name_scope("Infer_Decoder"):
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.decoder_embeddings, 
                                                            start_tokens=tf.fill([self._batch_size],
                                                            self._starting_string),
                                                            end_token = self._terminating_string)        
            decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, None)        
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                            impute_finished=True,
                                                            maximum_iterations=self.nr_steps_dec)
            outputs = tf.reshape(outputs.rnn_output, [-1, self._hidden_size])
        return outputs

    def fully_connected(self, decoder_output):
        W = tf.get_variable("Weights_Fully_Connected_Layer",dtype=tf.float32, shape=[self._hidden_size, self._vocab_size])
        b = tf.get_variable("Biases_Fully_Connected_Layer",dtype=tf.float32, shape=[self._vocab_size])
        return tf.matmul(decoder_output, W) + b

    def train(self):
        print("=========TRAIN\n\n")
        self.input_encoder = self.in_enc
        self.input_decoder = self.in_dec
        self.nr_steps_enc = self.nr_steps_in
        self.nr_steps_dec = self.nr_steps_out
        self.nr_epochs = 20 * self._training_slice
        return {
            "cost": self._cost,
            "train_op": self._train_op,
            "input_enc": self.input_encoder,
            "input_decoder": self.input_decoder
        }

    def valdiate(self):
        print("=========VALIDATION\n\n")
        self._data_slice = self._validation_slice
        self.input_encoder = self.in_enc
        self.input_decoder = self.in_dec
        self.nr_steps_enc = self.nr_steps_in
        self.nr_steps_dec = self.nr_steps_out
        return {
            "cost": self._cost,
            "input_enc": self.input_encoder,
            "input_decoder": self.input_decoder
        }

    def test(self):
        print("=========TEST\n\n")
        self._data_slice = self._test_slice
        self.input_encoder = self.test_enc
        self.input_decoder = self.test_dec
        self.nr_steps_enc = self.test_nr_steps_i
        self.nr_steps_dec = self.test_nr_steps_o
        self.nr_epochs = 2 * self._test_slice
        return {
            "cost": self._cost,
            "input_enc": self.input_encoder,
            "input_decoder": self.input_decoder
        }

    def run_model(self, operation, verbose=False):
        #self._batch_nr * self._data_slice)// self._epoch_size):
        for e in range(int(self.nr_epochs)):
            start_time = time.time()
            costs = 0.0
            iters = 0

            print("=========Epoch : %d" % e)
            for step in range(self._epoch_size):
                vals, summary = self.ses.run([operation,self.merged])
                costs += vals["cost"]
                iters += self.nr_steps_dec / 2
                if step == 0:
                    print(vals["input_enc"][0])
                    print("\n\n\n")

                if verbose and step % (self._epoch_size/10) == 0:
                    print("Nr : %.3f perplexity: %.3f speed: %.0f wps" %
                        (step * 1.0 / self._epoch_size, np.exp(costs / iters),
                         iters * self._batch_size /
                        (time.time() - start_time)))
                self.writer.add_summary(summary)

    def translate(self, sentence):
        sentence = [self.r.train_vocab[word] for word in sentence.replace(':', " :").replace('.', " .")
                                        .replace(',', " ,").replace(';', " ;").replace("\n", " <eos>").split(sep=' ')]
        self.input_encoder , in_enc = [sentence, self.input_encoder]
        self.batch_size = 1
        self.nr_steps_dec = len(sentence)
        output = self.ses.run([self.output])
        print(output)
        print(in_enc)

        print(sentence)

    def initialize_variables(self):
        self.ses.run(tf.global_variables_initializer())

    def tensordboard_write(self):
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./graphs', self.ses.graph)

    def create_saver(self):
        self.saver = tf.train.Saver()

    def save_model(self, path):
        self.saver.save(self.ses, path)

    def restore_model(self, path):
        self.saver.restore(self.ses, path)
        
    def close_translator(self):
        self.r.free_threads()
        self.ses.close()

    @property
    def batch_nr(self):
        return self._batch_nr

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def epoch_size(self):
        return self._epoch_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def layer_nr(self):
        return self._layer_nr

    @property
    def is_training(self):
        return self._is_training
    
    @property
    def config_training(self):
        self._is_training = True
    
    @property
    def config_testing(self):
        self._is_training = False

    @property
    def vocab_size(self):
        return self._vocab_size

class NormalConfig(object):
    in_file = in_file
    out_file = out_file
    layer_nr = 4
    batch_size = 25
    vocab_size = 100000
    keep_prob = 1
    hidden_size = 150
    max_grad_norm = 5
    epoch_size = 40
    training_slice = 1
    validation_slice = 0.05
    test_slice = 1


class LargeConfig(object):
    in_file = in_file
    out_file = out_file
    layer_nr = 4
    batch_size = 25
    vocab_size = 100000
    keep_prob = 0.7
    hidden_size = 150
    max_grad_norm = 5
    epoch_size = 100
    training_slice = 1
    validation_slice = 0.05
    test_slice = 1

t = Translator(NormalConfig())
#%%
# Testing
def main(_):
    t.model()
    t.initialize_variables()
    t.tensordboard_write()
    t.run_model(t.train(), verbose=True)
    t.create_saver()
    t.save_model("/tmp/model.ckpt")

    t.config_testing
    t.model()
    t.tensordboard_write()
    t.run_model(t.test(), verbose=True)

    t.close_translator()
    return

if __name__ == "__main__":
    tf.app.run()


#Try the model:

# t.create_saver()
# t.restore_model("/tmp/model.ckpt")
# print("Model restored")
# t.translate("<sos> Communication of Council common positions: see Minutes <eos>")


#%%
