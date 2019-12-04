
import os
import keras
from config import args
from layer_crf import ChainCRF
from keras.layers import TimeDistributed, Dense, Input, Embedding, GRU, Bidirectional
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import logging
import load_data
import tensorflow as tf
from keras_albert_model import load_brightmart_albert_zh_checkpoint, get_custom_objects



def config_bert_ner():
    bert_model = load_brightmart_albert_zh_checkpoint(args.bert_ckpt_dir, seq_len=128)
    bert_model.summary()
    sop_dense = bert_model.get_layer('MLM-Norm').output

    ner_dense = TimeDistributed(Dense(len(load_data.WS_TAGS)))(sop_dense)
    crf_layer = ChainCRF()
    crf_pred = crf_layer(ner_dense)
    loss = crf_layer.sparse_loss
    adam = keras.optimizers.Adam(lr=0.001)

    model = Model(inputs=bert_model.inputs, outputs=[crf_pred])
    model.compile(optimizer=adam, loss=loss, metrics=['sparse_categorical_accuracy'])
    model.summary(150)

    return model
import numpy as np
from layer_crf import ChainCRF
#model = config_bert_ner()
crf_layer = ChainCRF()
#crf_pred = crf_layer(ner_dense)
y1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12]).reshape((2,3,2))
print(y1)
loss = crf_layer.sparse_loss(y1, y1)
print(loss)