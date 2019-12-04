# encoding:utf-8
import sys
import os
import keras
from config import args
from layer_crf import ChainCRF
from keras.layers import TimeDistributed, Dense, Bidirectional, GRU, Activation
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from datetime import datetime
import numpy as np
import load_data
from keras_albert_model import load_brightmart_albert_zh_checkpoint, get_custom_objects

from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths, load_trained_model_from_checkpoint

'''
pip install keras-bert
pip install git+https://github.com/TinkerMob/keras_albert_model.git
'''
'''
model_path = get_pretrained(PretrainedList.chinese_base)
paths = get_checkpoint_paths(model_path)
print(paths.config, paths.checkpoint, paths.vocab)
'''
def config_bert_ner():
    classes = len(load_data.WS_TAGS)
    albert_model = load_brightmart_albert_zh_checkpoint(args.bert_ckpt_dir, seq_len=args.maxlen)
    #bert_model = load_trained_model_from_checkpoint(config_file="./res_models/chinese_L-12_H-768_A-12/bert_config.json",
    #                                                checkpoint_file="./res_models/chinese_L-12_H-768_A-12/bert_model.ckpt", seq_len=64)

    bert_model = albert_model
    bert_model.summary()
    bert_emb = bert_model.get_layer('MLM-Norm').output
    #bert_emb = bert_model.output

    birnn = Bidirectional(GRU(64, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(bert_emb)
    ner_dense = TimeDistributed(Dense(classes))(birnn)

    use_crf = False
    if use_crf:
        crf_layer = ChainCRF()
        pred = crf_layer(ner_dense)
        loss = crf_layer.sparse_loss
    else:
        pred = Activation('softmax')(ner_dense)
        loss = 'sparse_categorical_crossentropy'
    acc = 'sparse_categorical_accuracy'
    adam = keras.optimizers.Adam(lr=0.001, amsgrad=True)

    model = Model(inputs=bert_model.inputs, outputs=[pred])
    model.compile(optimizer=adam, loss=loss, metrics=[acc])
    model.summary(150)


    return model

def train_pipline():
    now = datetime.now().strftime("bert_ner-%Y.%m.%d_%H.%M.%S")
    if not os.path.exists(os.path.join(args.modeldir, now)):
        os.makedirs(os.path.join(args.modeldir, now))

    savepath = os.path.join(args.modeldir, now,
                            'e-{epoch:03d}-vl-{val_loss:.4f}-va-{val_sparse_categorical_accuracy:.4f}.h5')
    checkpointer = ModelCheckpoint(filepath=savepath,
                                   monitor='val_sparse_categorical_accuracy',
                                   mode='max', verbose=1, save_best_only=True)
    data_gener = load_data.DataGenerator(mode=1)

    train_line_indices, train_line_segments, train_line_tags, train_line_chs = data_gener.get_train()
    val_line_indices, val_line_segments, val_line_tags, val_line_chs = data_gener.get_dev()

    bert_downflow_model = config_bert_ner()

    x1, x2, y1 = np.array(train_line_indices), np.array(train_line_segments), np.expand_dims(np.array(train_line_tags), -1)
    print(x1.shape, x2.shape, y1.shape)
    v1, v2, w1 = np.array(val_line_indices), np.array(val_line_segments), np.expand_dims(np.array(val_line_tags), -1)
    print(v1.shape, v2.shape, w1.shape)
    bert_downflow_model.fit([x1, x2], y1, batch_size=16,
                            epochs=50,  verbose=1, callbacks=[checkpointer],
                            validation_data=([v1, v2], w1), shuffle=True, )


def test_pipline():
    custom_objects = {"ChainCRF": ChainCRF}
    custom_objects.update(get_custom_objects())
    model = load_model(filepath=args.ckpt, custom_objects={"ChainCRF": ChainCRF}, compile=False)
    data_gener = load_data.DataGenerator(mode=1)
    val_line_indices, val_line_segments, val_line_tags, val_line_chs = data_gener.get_dev()

    pred_tags = model.predict([val_line_indices, val_line_segments])
    if pred_tags.shape[-1] > 1:
        pred_class = pred_tags.argmax(axis=-1)
    else:
        pred_class = (pred_tags > 0.5).astype('int32')
    TAG = {0: 'B', 1: 'I', 2: 'E', 3: 'S'}
    result = args.pred_result.open('w', encoding='utf-8')
    for i, chs in enumerate(val_line_chs):
        new_line = ""
        for j, t in enumerate(pred_class[i]):
            if j == 0: continue
            if chs[j] == "[SEP]": break
            new_line += chs[j]

            if t in [2, 3]:
                new_line += ' '
        result.write(new_line + "\n")
    result.flush()
    result.close()


def pipline():
    if args.do_train:
        train_pipline()
    if args.do_test:
        test_pipline()


if __name__ == "__main__":
    pipline()