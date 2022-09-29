from distutils.log import debug
from tensorflow import keras
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv2D, MaxPool2D, Flatten, Conv1D, LSTM, InputLayer
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
import collections
import nltk
import math
import requests
import json
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize 
from nltk.tokenize.treebank import TreebankWordDetokenizer #to form sentences form tokens wherever needed.
from nltk.stem.snowball import SnowballStemmer
import itertools
import torch
import tensorflow as tf
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import pipeline
from pattern.en import conjugate ,pluralize, singularize
from pattern.en import tag
import gensim
import gensim.downloader as api
from collections import OrderedDict
import random
from random import choice
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows




tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# # from transformers import BertTokenizer, BertModel
# # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# model = BertModel.from_pretrained("bert-base-cased")
def Tokenize(txt):
    marked_text = "[CLS] " + txt + " [SEP]"

    tokenized_text = tokenizer.tokenize(marked_text)
    return tokenized_text

def index_tokens(txt):
    indexed_tokens = tokenizer.convert_tokens_to_ids(txt)
    return indexed_tokens

def segment_ids(txt):
    
    segments_ids = [1] * len(txt)
    return segments_ids

# Convert inputs to PyTorch tensors
def tensor_maker(txt):
    tokens_tensor = torch.tensor([txt])
    return tokens_tensor
tf.compat.v1.enable_eager_execution()
# Load pre-trained model (weights)
model = BertModel.from_pretrained("bert-base-uncased",
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

def get_bert_embeddings(tokens_tensor, segments_tensor, model):
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensor)
    return outputs[2]
def Club_words_forcnn(tokens, embeddings):
    new_tok = []
    new_embed = []

    i=0
    while i<=len(tokens)-1:
        curr = tokens[i]
        vec = embeddings[i,:,:]
        count = 1

        for j in range(i+1, len(tokens)):
            if(tokens[j][:2]=='##'):
                curr+=tokens[j][2:]
                vec+=embeddings[j,:,:]
                count+=1

            else:
                i=j-1
                break
        i+=1
        vec = vec/count
        new_tok.append(curr)
        new_embed.append(vec)


    vec = torch.stack(new_embed)
        
    return new_tok, vec
def get_shaped_embed(vec, num):
    # layers = []
    # for i in range(num):
    #     layers.append(vec[i][0])
    # for i in range(25-num,25):
    #     layers.append(vec[i][0])

    # temp = torch.reshape(layers[0][0,:],(1,1024))
    # for i in range(1,2*num):
    #     temp = torch.cat((temp, torch.reshape(layers[i][0,:],(1,1024))), dim=0)
        
    # temp = torch.reshape(temp, (1,2*num,1024))


    # for i in range(1, len(layers[0])):

    #     tmp = torch.reshape(layers[0][i,:],(1,1024))
    #     for j in range(1,2*num):
    #         tmp = torch.cat((tmp, torch.reshape(layers[j][i,:],(1,1024))), dim=0)
    #     tmp = torch.reshape(tmp, (1,2*num,1024))

    #     temp = torch.cat((temp, tmp), dim=0)
    layers = []
    for i in range(1,num+1):
        layers.append(vec[i][0])
    for i in range(13-num,13):
        layers.append(vec[i][0])

    temp = torch.reshape(layers[0][0,:],(1,768))
    for i in range(1,2*num):
        temp = torch.cat((temp, torch.reshape(layers[i][0,:],(1,768))), dim=0)
        
    temp = torch.reshape(temp, (1,2*num,768))


    for i in range(1, len(layers[0])):

        tmp = torch.reshape(layers[0][i,:],(1,768))
        for j in range(1,2*num):
            tmp = torch.cat((tmp, torch.reshape(layers[j][i,:],(1,768))), dim=0)
        tmp = torch.reshape(tmp, (1,2*num,768))

        temp = torch.cat((temp, tmp), dim=0)
    return temp

clf_model12 =tf.keras.models.load_model('content\CWI_all_lstm_gen_v12')
def CWI(sent, bert_model, clf_model):
    tok = Tokenize(sent)
    ind_tok = index_tokens(tok)
    seg_tok = segment_ids(tok)

    ind_tok = tensor_maker(ind_tok)
    seg_tok = tensor_maker(seg_tok)

    vec = get_bert_embeddings(ind_tok, seg_tok, bert_model) 

    vec = get_shaped_embed(vec, 6)

    new_tok, x = Club_words_forcnn(tok, vec)

    x = x.numpy()

    pred = clf_model.predict(x)
    max1=0
    for i in range(len(new_tok)):
        # print(f'Token: {new_tok[i]}      Complexity: {pred[i]}')
        if(pred[i] >= max1):
          tok = new_tok[i] 
          max1 = pred[i]
    return tok
