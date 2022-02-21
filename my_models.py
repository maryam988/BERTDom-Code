import os

import pandas as pd
import numpy as np

import re

import tensorflow as tf
from tensorflow import keras
from  tensorflow.keras.callbacks import EarlyStopping

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

def create_model_dense_dropout(max_seq_len, bert_ckpt_file, bert_config_file):
    # https://www.curiousily.com/posts/intent-recognition-with-bert-using-keras-and-tensorflow-2/
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name="bert")
    
    input_ids = keras.layers.Input(
        shape=(max_seq_len, ),
        dtype='int32',
        name="input_ids"
    )
    bert_output = bert(input_ids)
    
    print("bert shape", bert_output.shape)

    classes = 200

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=512, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(
        units=classes,
        activation="sigmoid"
    )(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    

    load_stock_weights(bert, bert_ckpt_file)
    model.build(input_shape=(None, max_seq_len))

    return model

def create_model_bilstm(max_seq_len, bert_ckpt_file, bert_config_file):

  with tf.io.gfile.GFile(bert_config_file, "r") as reader:
      bc = StockBertConfig.from_json_string(reader.read())
      bert_params = map_stock_config_to_params(bc)
      bert_params.adapter_size = None
      bert = BertModelLayer.from_params(bert_params, name="bert")
        
  input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
  bert_output = bert(input_ids)

  print("bert shape", bert_output.shape)

  classes = 200

  x1 = keras.layers.Bidirectional(keras.layers.LSTM(100,return_sequences=True),merge_mode='sum')(bert_output)
#   x4 = keras.layers.Bidirectional(keras.layers.LSTM(3,return_sequences=True),merge_mode='sum')(bert_output)
    
  x2 = keras.layers.Bidirectional(keras.layers.LSTM(100,return_sequences=True),merge_mode='sum')(x1)

  x3 = keras.layers.Bidirectional(keras.layers.LSTM(100,return_sequences=True),merge_mode='sum')(x2)
  x4 = keras.layers.Bidirectional(keras.layers.LSTM(3,return_sequences=True),merge_mode='sum')(x3)

  output = keras.layers.Activation('softmax')(x4)


#   cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
#   cls_out = keras.layers.Dropout(0.5)(cls_out)
#   logits = keras.layers.Dense(units=512, activation="tanh")(cls_out)
#   logits = keras.layers.Dropout(0.5)(logits)
#   logits = keras.layers.Dense(units=classes, activation="sigmoid")(logits)

  model = keras.Model(inputs=input_ids, outputs=output)
  

  load_stock_weights(bert, bert_ckpt_file)
  model.build(input_shape=(None, max_seq_len))
        
  return model