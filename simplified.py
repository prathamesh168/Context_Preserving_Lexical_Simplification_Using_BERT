# to form sentences form tokens wherever needed.
from CWI import *
from pattern.en import conjugate, pluralize, singularize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize
from random import choice
import gensim.downloader as api
from pattern.en import tag
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import tensorflow as tf
import torch
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import numpy as np
import nltk
import requests
import json
# nltk.download('punkt')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
model_gigaword = api.load("glove-wiki-gigaword-300")
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased',)

mlm_model.eval()


def Bert_pred(sent, c_w):  # need the index in terms of nltk tokenizer, make sure to pass that way
    tokens = word_tokenize(sent)
    sec_tok = tokens.copy()

    index = tokens.index(c_w)
    tokens[index] = "[MASK]"

    threshold = 0.1  # mask some percent of the sentence
    num = int(threshold*len(tokens))
    ps = 0

    # for epochs in range(10):
    # sec_tok = tokens.copy()

    # rand_inds = np.random.randint(low=0, high=len(tokens), size=num, dtype=int)
    # rand_inds = np.unique(rand_inds)
    for epochs in range(5):
        sec_tok = word_tokenize(sent)

        rand_inds = []
        for i in range(num):
            rand_inds.append(
                choice([j for j in range(0, len(tokens)) if j not in [index]]))
        rand_inds = np.array(rand_inds)
        rand_inds = np.unique(rand_inds)
        for i in rand_inds:
            sec_tok[i] = '[MASK]'

        sec_tok.append('[SEP]')
        sec_tok.insert(0, '[CLS]')
        # print(sec_tok)
        fin_tok = sec_tok+tokens
        sentence = TreebankWordDetokenizer().detokenize(fin_tok)

        fin_tok = tokenizer.tokenize(sentence)

        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(fin_tok)])

        with torch.no_grad():
            output = mlm_model(tensor_input).logits

        half_sent = tokens[:index]
        half_sent = TreebankWordDetokenizer().detokenize(half_sent)
        half_sent = tokenizer.tokenize(half_sent)
        full_sent = TreebankWordDetokenizer().detokenize(sec_tok)
        full_sent = tokenizer.tokenize(full_sent)
        # print(fin_tok)
        k = len(full_sent) + len(half_sent)
        mask_out = output[0, k, :]
        softmax = torch.nn.Softmax(dim=0)
        ps += softmax(mask_out)
    ps = ps/5
    sorted_arr = np.argsort(ps)

    decode = tokenizer.decode(sorted_arr[-20:])
    # print(word_tokenize(decode))
    return word_tokenize(decode), ps[sorted_arr]

def cos_sim(a, b):

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)
def get_embeds(c_w,sent,bert_model):
    tokens = word_tokenize(sent)
    ind = tokens.index(c_w)

    tok = Tokenize(sent)
    ind_tok = index_tokens(tok)
    seg_tok = segment_ids(tok)

    ind_tok = tensor_maker(ind_tok)
    seg_tok = tensor_maker(seg_tok)

    vec = get_bert_embeddings(ind_tok, seg_tok, bert_model)
    vec = vec[24][0]
    return vec[ind,:]


def bert_embed_sim(c_w, synonyms, sent,model):
    vec1 = get_embeds( c_w,sent,model)

    tokens = word_tokenize(sent)
    
    ind = tokens.index(c_w)
    sim = {}

    for s_w in synonyms:
        tokens[ind] = s_w

        simp_sent = TreebankWordDetokenizer().detokenize(tokens)
        print(simp_sent)
        vec2 = get_embeds(s_w,simp_sent, model)
        sim[s_w] = cos_sim(vec1, vec2)
    # print(sim)

    return sim


def get_embedds( index, sent,model):
    tokens = word_tokenize(sent)

    tok = Tokenize(sent)
    ind_tok = index_tokens(tok)
    seg_tok = segment_ids(tok)

    ind_tok = tensor_maker(ind_tok)
    seg_tok = tensor_maker(seg_tok)

    vec = get_bert_embeddings(ind_tok, seg_tok, model)
    vec = vec[24][0]
    return vec[index,:]
def context_sim(c_w, synonyms, sent, model):
    tokens = word_tokenize(sent)
    
    ind = tokens.index(c_w)
    sim = {}
    
    for s_w in synonyms:
        sim[s_w] = 0

    for i in range(-5,6):
    
        index = i+ind  #offset index
        if index<0 or index>=len(tokens) or tokens[index]=='[CLS]' or tokens[index]=='[SEP]' or i==0: 
            continue
        
        vec1 = get_embedds( index, sent,model)
        
        for s_w in synonyms:
            tokens[ind] = s_w

            simp_sent = TreebankWordDetokenizer().detokenize(tokens)     
            vec2 = get_embedds( index, simp_sent, model)

            cos = cos_sim(vec1, vec2)
            sim[s_w]+=(cos/10)
    # print(sim)
    return sim



tag_corr = {
    'VB': 'inf',
    'VBP':'1sg',
    'VBZ':'3sg',
    'VBG':'part',
    'VBD':'p',
    'VBN':'ppart',
}

#Semantic Similarity


def synonyms_f(sent,cw, synonyms):
    # synonyms,ps = Bert_pred(sent,cw,index)
    # print(synonyms)
    syn_rank= {}
    cw_tag = tag(cw)
    # print(cw_tag[0][1])
    for i in synonyms:
        # print(conjugate(i,'1sg'))
        # print(singularize(i))
        tag_word= tag(i)
        # print(tag_word[0][1])
        gamma =0
        if  i != cw :
          # print(i)
          if (tag_word[0][1] != "NN" or tag_word[0][1] != "NNS" or tag_word[0][1] != "JJ" or tag_word[0][1] != "JJR" or tag_word[0][1] != "JJS") and conjugate(i,'1sg') != conjugate(cw,'1sg') and cw_tag[0][1] in tag_corr and tag_word[0][1] in tag_corr:
              i = conjugate(i,tag_corr[cw_tag[0][1]])
              # print(g)
              try:
                gamma = model_gigaword.wv.similarity(cw,i)
              except:
                gamma = 0
          
          elif (singularize(cw) != singularize(i) ) and cw_tag[0][1] == tag_word[0][1] :
              # if model_gigaword.wv.similarity(cw,i):
            try:
                gamma = model_gigaword.wv.similarity(cw,i)
            except:
                gamma =0
          if gamma >= .2:
            syn_rank[i] = gamma 
    # print(syn_rank)
    dict3={}
    # for i in range(len(syn_rank)):
        # dict1 = dict(itertools.islice(replaced_w[mask[i]].items()))
    dict3 = dict(sorted(syn_rank.items(), key=lambda item: item[1]))
    # print(dict3)
    return dict3


# Cross-Entropy
def cross_entropy_word(X,i,pos):
    #print(X) 
    #print(X[0,2,3])
    softmax = torch.nn.Softmax(dim=0)
    X = softmax(X) 
    loss = 0 
    loss -= np.log10(X[i,pos]) 
    return -loss


### random masking and predicting
def masking(sent, s_w):
    # sent1 = sent #"John wrote these verses."
    index = word_tokenize(sent).index(s_w)

    sent1 = "[CLS] " + sent + " [SEP]"
    # sent2 = sent1[0:] 
    # print(sent1)
    sentence_loss = 0
    tmp = word_tokenize(sent1)
    # input_ids = tokenizer.convert_tokens_to_ids(tmp)
    # tmp_tok1 = tokenizer.tokenize(sent2)
    for i in range(-5,6): 
        if(i==0 or i+index<0 or i+index>len(tmp)-1):
            continue
        if tmp[i+index] == "[CLS]" or tmp[i] == "[SEP]":
            continue
        # tmp_tok1[index_cw] = "[MASK]"
        # tmp[random.randint(1,len(tmp)-2)] = "[MASK]"
        sw = tmp[i+index]
        tmp[i+index] = "[MASK]"
        tmp_sent = TreebankWordDetokenizer().detokenize(tmp)
        tmp_tok = tokenizer.tokenize(tmp_sent) 

        input_ids = tokenizer.convert_tokens_to_ids(tmp_tok)
        # print(tmp_tok)   
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tmp_tok)])
        with torch.no_grad():
            output = mlm_model(tensor_input).logits
        # mask_out = output[0,len(tmp_tok1)+index_cw,:]
        
        # ps = softmax(mask_out)
        # top_10 = np.argpartition(ps,-10)[-10:]
        # decode = tokenizer.decode(top_10)
        # print(decode)
        word_loss = cross_entropy_word(output[0],i+index,input_ids[i+index])
        sentence_loss += word_loss
        tmp[i+index] = sw
    # print(np.exp(sentence_loss/len(tmp)))
    return np.exp(sentence_loss/len(tmp))


def cross_sent(sent,clf_model,model,synonyms,mask):  
  # sent = 'John composed these verses.'
  sent_token = nltk.tokenize.word_tokenize(sent.lower())
  # print(sent_token)
  # mask,mask_max,max4 = Pipeline2(sent,clf_model,model)
  for i in range(len(sent_token)):
    if (mask == sent_token[i]):
      mask_ind = i
  mask_con = {}
  # mask_ind = mask_max -1
  
#   print(mask_ind)
  
  mask_con[mask] = list(synonyms_f(sent,mask,synonyms).keys())
  # print(mask_con[mask[0]])
  j=0
  replaced_w = {}
  
    
  dict_w_ce = {}
  for k in range(len(mask_con[mask])):
    sent_token[mask_ind] = mask_con[mask][k]
    sent_join = ' '.join(sent_token)
    cross_entropy = masking(sent_join, mask_con[mask][k])
    dict_w_ce[mask_con[mask][k]] = cross_entropy
    # print(dict_w_ce)
    replaced_w[mask] = dict_w_ce
#   print(replaced_w)
  dict2 = {}
  for i in range(len(mask)):
    # dict1 = dict(itertools.islice(replaced_w[mask[i]].items()))
    dict2 = dict(sorted(replaced_w[mask].items(), key=lambda item: item[1]))
    # print(dict2)
  return dict2

  threshold = 0.01

def Pipeline2(sent, clf_model,bert_model):
    
    tok = Tokenize(sent)
    ind_tok = index_tokens(tok)
    seg_tok = segment_ids(tok)

    ind_tok = tensor_maker(ind_tok) 
    seg_tok = tensor_maker(seg_tok)

    vec = get_bert_embeddings(ind_tok, seg_tok,bert_model) 

    vec = get_shaped_embed(vec, 5)

    new_tok, x = Club_words_forcnn(tok, vec)

    x = x.numpy()

    # x = x.reshape((len(new_tok),4,768))
    pred = clf_model.predict(x)
    max1 = 0
    maxi = 0
    tok = " "
    index = []

    for i in range(len(new_tok)):
        print(f'Token: {new_tok[i]}      Complexity: {pred[i]}')
        if(pred[i] >= max1):
          tok = new_tok[i]
          maxi = i 
          max1 = pred[i]
    return tok,maxi,max1


def Pipeline1(sent,cw, clf_model,bert_model):
    tok = Tokenize(sent)
    ind_tok = index_tokens(tok)
    seg_tok = segment_ids(tok)

    ind_tok = tensor_maker(ind_tok) 
    seg_tok = tensor_maker(seg_tok)

    vec = get_bert_embeddings(ind_tok, seg_tok,bert_model) 

    vec = get_shaped_embed(vec, 5)

    new_tok, x = Club_words_forcnn(tok, vec)

    x = x.numpy()
    # x = x.reshape((len(new_tok),4, 768))
    
    pred = clf_model.predict(x)
    new_complex = pred[new_tok.index(cw)]
    # print(new_tok[ind_complex+1])
    return new_complex

def Complexity(sent,clf_model,model,synonyms,mask):  
  # sent = 'John composed these verses.'
  sent_token = nltk.tokenize.word_tokenize(sent.lower())
  # # print(sent_token)
  # for i in range(len(sent_token)):
  #   if (mask == sent_token[i]):
  mask_ind = sent_token.index(mask)
  
  mask_con = {}
  max3 = Pipeline1(sent,mask,clf_model,model)
  # mask_ind = mask_max-1
  # print(max3)
  # print(sent_token)

  # print(mask_ind)
  
  mask_con[mask] = synonyms#list(synonyms_f(sent,mask,synonyms).keys()) 
  # print(mask_con[mask])
  j=0
  replaced_w = {}
  
    
  dict_w_ce = {}

  for k in range(len(mask_con[mask])):
    try: 
      sent_token[mask_ind] = mask_con[mask][k]
      sent_join = TreebankWordDetokenizer().detokenize(sent_token)     
      # print(sent_join)
      complexity = Pipeline1(sent_join,mask_con[mask][k],clf_model,model)
      # print(complexity)
      if max3 >= complexity:
        
        dict_w_ce[mask_con[mask][k]] = complexity
    #   print(dict_w_ce)
        replaced_w[mask] = dict_w_ce
    except:
      print(" ")
  # Creates a sorted dictionary (sorted by key)
  from collections import OrderedDict
  import itertools
  dict2 = {}
  for i in range(len(mask)):
    # dict1 = dict(itertools.islice(replaced_w[mask[i]].items()))
    try:
      dict2 = dict(sorted(replaced_w[mask].items(), key=lambda item: item[1]))
    except:
      print(" ")
  #print(dict2)
  return dict2

key = '8514d7ebcbd22a14867c78748bf40d1f'
stemmer = SnowballStemmer("english")
# synonym finiding wit big thesaurus and complexity checking 
from pattern.en import tag
def conversion(word_con):
  modified_word = []
  pb = word_con
  tag_word= tag(pb)
  tag1 = tag_word[0][1]
#   print(tag1)
  if tag1 in tag_corr.keys():
    root_w = conjugate(pb,'1sg')
  elif tag1 == "NNS":
    # root_w= stemmer.stem(pb)
    root_w = singularize(pb)
  elif tag1 == "JJ" or tag1 == "JJR" or tag1 == "JJS":
    root_w = pb   
  else:
    root_w = pb
#   print(root_w)

  req =requests.get('https://words.bighugelabs.com/api/2/'+key+'/'+root_w+'/json')
  req1 = req.text
  try:
    data = json.loads(req1)
  #   print(data)
    if data:
      if type(data) == dict:
        if tag1 == 'NN' or tag1 == 'NNS':
          data1 = data['noun']
        elif tag1 in tag_corr.keys():
          data1 = data['verb']
        elif tag1== "JJ" or tag1 =="JJR" or tag1 == "JJS":
          data1 = data['adjective']
        for i in range(len(data1['syn'])): 
          if  tag1 in tag_corr.keys():
            word = conjugate(data1['syn'][i],tag_corr[tag1])
          elif tag1 == 'NNS':
            word = pluralize(data1['syn'][i])
          elif tag1 == 'NN':
            word = data1['syn'][i]
          elif tag1== "JJ" or tag1 =="JJR" or tag1 == "JJS":
            word = data1['syn'][i]  
          
          modified_word.append(word)
      elif type(data) == list:
        for i in range(len(data)): 
          if tag1 in tag_corr.keys():
            word = conjugate(data[i],tag_corr[tag1])
          elif tag1 == 'NNS':
            word = pluralize(data[i])
          elif tag1 == 'NN':
            word = data[i]
          elif tag1== "JJ" or tag1 =="JJR" or tag1 == "JJS":
            word = data1[i]  
            
          modified_word.append(word)
  except:
      print("")
  return modified_word

import string
def get_score(sent,clf_model,c_w):
  # c_w,index_cw,max8 = Pipeline2(sent,clf_model,model)
  meth1,prob = Bert_pred(sent,c_w)
  prob = prob.numpy().tolist()
  i=0
  while (i<len(meth1)):
    if meth1[i].translate(str.maketrans('', '', string.punctuation)) == "":
      meth1.pop(i)
      prob.pop(i)
    i+=1

  # print(meth1)
#   synonyms = conversion(c_w)
#   i=0
#   while(i<len(synonyms)):
#     if " " in synonyms[i] or "-" in synonyms[i]:
#       synonyms.pop(i)
#     i+=1
#   meth1 = meth1 + synonyms
#   meth2 = cross_sent(sent,clf_model,model,meth1,c_w)
  # print(meth1)
  meth3 = synonyms_f(sent,c_w,meth1)
  meth4 = Complexity(sent,clf_model,model,list(meth3.keys()),c_w)
  meth5 = context_sim(c_w,list(meth4.keys()),sent,model)
  score = {i: meth5[i] for i in meth5}
  score1 = score
  # print(score)
  # print(meth1)
  
  # print(meth3)
  # print(meth4)
  # print(meth5)
  for i in score.keys():
    # if i in meth3.keys()  :
#       score[i] +=   meth5[i]

#     #   score[i] -= meth4[i][0]

    score[i]+= meth3[i]

  # print(score )  
  res = dict(sorted(score.items(), key=lambda item: item[1],reverse=True)) 
  #if score1 != score:
  # print(res) 
  #else:
    #print("no simplification needed.")
  return list(res.keys())


print(get_score("cat perched on the mat.",clf_model12,CWI("cat perched on the mat.",model,clf_model12)))