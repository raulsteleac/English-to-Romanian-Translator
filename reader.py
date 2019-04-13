#%%
import os
import numpy as np
import tensorflow as tf
import collections
#%%
in_file = "/home/raulslab/work/e_to_r_translation/data/europarl-v7.ro-en.en"
out_file = "/home/raulslab/work/e_to_r_translation/data/europarl-v7.ro-en.ro"
vocab_size = 100000


def with_prefix(path, file):
        return "/".join((path, file))


class Reader(object):    
    def __init__(self, in_file, out_file, vocab_size, batch_size):
        self._in_file = in_file
        self._out_file = out_file
        self._vocab_size = vocab_size
        self._batch_size = batch_size

    def import_data_vocab(self, file_name):
        with tf.gfile.Open(file_name, 'r') as f:
                return f.read().replace(':'," :").replace('.'," .").replace(','," ,").replace(';'," ;").replace("\n"," <eos> <sos>").split(sep=' ')
    
    def import_data(self, file_name):
        with tf.gfile.Open(file_name, 'r') as f:
            data = f.readlines()
            data = [("<sos> " + line).replace(':'," :").replace('.'," .").replace(','," ,").replace(';'," ;").replace("\n"," <eos>").split(sep=' ') for line in data]
            print("-----------Importing data from file")
            return data[0:40001], data[40001:44001]

    def build_vocab(self, file_name):
        print("-----------Building Vocab")
        data = self.import_data_vocab(file_name)
        count = collections.Counter(data)
        count = sorted(count.items(), key=lambda x:(-x[1], x[0]))
        cuv, _ = list(zip(*count))
        cuv = cuv[0:vocab_size]
        self._word_to_id = dict(zip(cuv, [i for i in range(len(cuv))]))        

    def get_words_to_ids(self, data, vocab):
        return np.array([[vocab[cuv] for cuv in line if cuv in vocab] for line in data])

    def translator_raw_data(self, file_name):
        print("-----------Creating Translator like formated data")
        self.build_vocab(file_name)
        data , test= self.import_data(file_name)
        data = self.get_words_to_ids(data, self._word_to_id)
        test = self.get_words_to_ids(test, self._word_to_id)
        data_len = len(data)
        test_len = len(test)
        self._batch_nr = data_len // self._batch_size
        self.test_batch_nr = test_len // self._batch_size
        print("Batch nr train : %d " % self._batch_nr)
        print("Batch nr test  : %d" % self.test_batch_nr)

        nr_steps = self.compute_nume_steps_per_batch(data)
        test_nr_steps = self.compute_nume_steps_per_batch(test)

        data = data[0: self._batch_nr *
                  self._batch_size].reshape((self._batch_nr, self._batch_size))

        data = [self.reshape_lines(data[i], nr_steps)
                for i in range(self._batch_nr)]

        test = test[0: self.test_batch_nr *
                    self._batch_size].reshape((self.test_batch_nr, self._batch_size))
        
        test = [self.reshape_lines(test[i], test_nr_steps)
                for i in range(self.test_batch_nr)]

        return data, nr_steps, test, test_nr_steps

    def compute_nume_steps_per_batch(self, batch):
        return max([len(line) for line in batch])

    def reshape_lines(self, batch, nr_steps):
        lis = []
        for i in range(self._batch_size):
                z = np.array(batch[i])
                z.resize(nr_steps)
                lis.append([*z])
        return lis

    def translator_batch_producer(self, session, name=None):
        with tf.name_scope(name, "Translator_Data_Producer", [self._batch_size, session]):
            print("-----------Producing inputs")

            d_i, nr_steps_in, t_i, train_nr_steps_i = self.translator_raw_data(self._in_file)
            d_i = tf.convert_to_tensor(d_i)
            t_i = tf.convert_to_tensor(t_i)
            self.train_vocab = self._word_to_id

            d_o, nr_steps_out, t_o, test_nr_steps_o = self.translator_raw_data(self._out_file)
            d_o = tf.convert_to_tensor(d_o)        
            t_o = tf.convert_to_tensor(t_o)
            self.test_vocab = self._word_to_id

            print("-----------Converting raw data")

            queue = tf.FIFOQueue(capacity=self._batch_nr, dtypes=[tf.int32])
            enqueue_op = queue.enqueue_many([[j for j in range(self._batch_nr)]])
            i = queue.dequeue()

            self._coord = tf.train.Coordinator()
            qr = tf.train.QueueRunner(
                queue=queue, enqueue_ops=[enqueue_op] * 2)
            self._threads = qr.create_threads(session, self._coord, start=True)

            in_enc = d_i[i]
            in_dec = d_o[i]
            test_enc = t_i[i % self.test_batch_nr]
            test_dec = t_o[i % self.test_batch_nr]
            
            return in_enc, in_dec, nr_steps_in, nr_steps_out, test_enc, train_nr_steps_i, test_dec, test_nr_steps_o, self._word_to_id["<sos>"], self._word_to_id["<eos>"]

    def free_threads(self):
        self._coord.request_stop()
        self._coord.join(self._threads)

#%%
#Testing
# with tf.Session() as ses:
#         r = Reader(in_file=in_file, out_file=out_file, vocab_size=vocab_size, batch_size=100)
#         x, y, nr_steps_in, nr_steps_out,_,_,_,_ = r.translator_batch_producer(ses)
#         print(ses.run(x[0]))
#         print(ses.run(x[0]))
#         r.free_threads()


#%%


#%%
