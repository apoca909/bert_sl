#! -*- coding: utf-8 -*-

import json
from tqdm import tqdm
import os, re
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
from keras_contrib.layers.crf import CRF

mode = 0
maxlen = 128
learning_rate = 5e-5
min_learning_rate = 1e-5

config_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)

D = pd.read_csv('./data/event_type_entity_extract_train.csv', encoding='utf-8', header=None)
D = D[D[2] != u'其他']
classes = set(D[2].unique())

train_data = []
for t, c, n in zip(D[1], D[2], D[3]):
    train_data.append((t, c, n))

if not os.path.exists('./data/random_order_train.json'):
    random_order = [_ for _ in range(len(train_data))]
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('./data/random_order_train.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('./data/random_order_train.json'))

dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]
additional_chars = set()
for d in train_data + dev_data:
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', d[2]))

additional_chars.remove(u'，')

D = pd.read_csv('./data/event_type_entity_extract_eval.csv', encoding='utf-8', header=None)
test_data = []
for id, t, c in zip(D[0], D[1], D[2]):
    test_data.append((id, t, c))


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i + n_list2] == list2:
            return i
    return -1


class data_generator:
    def __init__(self, data, batch_size=8):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = [_ for _ in range(len(self.data))]
            np.random.shuffle(idxs)
            X1, X2, X3, S1, S2 = [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                text, c = d[0][:maxlen], d[1]
                text = u'___%s___%s' % (c, text)
                tokens = tokenizer.tokenize(text)
                e = d[2]
                e_tokens = tokenizer.tokenize(e)[1:-1]
                _s1, _s2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                start = list_find(tokens, e_tokens)
                if start != -1:
                    end = start + len(e_tokens) - 1
                    _s1[start] = 1
                    _s2[end] = 1
                    _x1, _x2 = tokenizer.encode(first=text)
                    _x3 = [1 for _ in _x1]
                    X1.append(_x1)
                    X2.append(_x2)
                    X3.append(_x3)
                    S1.append(_s1)
                    S2.append(_s2)
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        X3 = seq_padding(X3)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        yield [X1, X2, X3], [S1, S2]
                        X1, X2, X3, S1, S2 = [], [], [], [], []

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,), name='token_ids', dtype='int32')  # 待识别句子输入 token ids
x2_in = Input(shape=(None,), name='segment_ids', dtype='int32')  # 待识别句子输入 segment ids
x3_in = Input(shape=(None,), name='token_masks', dtype='float32')  # mask
# s1_in = Input(shape=(None,), name='subject_left') # 实体左边界（标签）
# s2_in = Input(shape=(None,), name='subject_right') # 实体右边界（标签）

x1, x2, x_mask = x1_in, x2_in, x3_in
# s1, s2 = s1_in, s2_in
x = bert_model([x1, x2])
x_mask = Lambda(lambda x: K.expand_dims(x, 2))(x_mask)

ps1 = Dense(1, use_bias=False)(x)
ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10, name='output_left')([ps1, x_mask]) #mask/补充位直接扔掉


ps2 = Dense(1, use_bias=False)(x)
ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10, name='output_right')([ps2, x_mask])

pred_model = Model(inputs=[x1_in, x2_in, x3_in], outputs=[ps1, ps2])

train_model = Model(inputs=[x1_in, x2_in, x3_in], outputs=[ps1, ps2])


def loss_left(y_true, y_pred):
    loss = K.mean(K.categorical_crossentropy(y_true, y_pred, from_logits=True))
    return loss

def loss_right(y_true, y_pred):
    #print(y_true_1.shape)
    #y_true_1_sum = K.cumsum(y_true, 1)
    #y_pred_2 = y_pred_2 - (1 - y_true_1_sum) * 1e10
    loss = K.mean(K.categorical_crossentropy(y_true, y_pred, from_logits=True))
    return loss


losses = {
    "output_left": loss_left,
    "output_right": loss_right
}

loss_weights = {
    "output_left": 1.0,
    "output_right": 1.0
}

train_model.compile(optimizer=Adam(learning_rate), loss=losses, loss_weights=loss_weights)
train_model.summary()


def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


def extract_entity(text_in, c_in):
    if c_in not in classes:
        return 'NaN'
    text_in = u'___%s___%s' % (c_in, text_in)
    text_in = text_in[:510]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x3 = [1 for _ in _x1]
    _x1, _x2, _x3 = np.array([_x1]), np.array([_x2]), np.array([_x3])
    _ps1, _ps2 = train_model.predict([_x1, _x2, _x3])
    _ps1, _ps2 = softmax(_ps1[0]), softmax(_ps2[0])
    for i, _t in enumerate(_tokens):
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            _ps1[i] -= 10
    start = _ps1.argmax()
    for end in range(start, len(_tokens)):
        _t = _tokens[end]
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            break
    end = _ps2[start:end + 1].argmax() + start
    a = text_in[start - 1: end]
    return a


class Evaluate(Callback):
    def __init__(self):
        self.ACC = []
        self.best = 0.
        self.passed = 0

    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            train_model.save_weights('best_model.weights')
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))

    def evaluate(self):
        A = 1e-10
        F = open('dev_pred.json', 'w')
        for d in tqdm(iter(dev_data)):
            R = extract_entity(d[0], d[1])
            if R == d[2]:
                A += 1
            s = ', '.join(d + (R,))
            F.write(s + '\n')
        F.close()
        return A / len(dev_data)


def test(test_data):
    F = open('result.txt', 'w')
    for d in tqdm(iter(test_data)):
        s = u'"%s","%s"\n' % (d[0], extract_entity(d[1], d[2]))
        s = s.encode('utf-8')
        F.write(s)
    F.close()


evaluator = Evaluate()
train_D = data_generator(train_data)

if __name__ == '__main__':
    train_model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=len(train_D),
                              epochs=10,
                              callbacks=[evaluator]
                              )
else:
    train_model.load_weights('best_model.weights')
