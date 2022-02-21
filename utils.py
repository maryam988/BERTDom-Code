import numpy as np
import re
# from bert.tokenization.bert_tokenization import FullTokenizer
from tqdm.notebook import tqdm as tqdm

import torch
# from tape import ProteinBertModel, TAPETokenizer

def process_inputseqs(file):
    prot_id = ''
    prot_seq = ''
    seqs = []
    ids = []
    for line in open(file, 'r'):
        if line[0] == '>':
            if prot_id != '':
                seqs.append(prot_seq)
                ids.append(prot_id)
            
            prot_id = line.strip()
            prot_seq = ''
        elif line.strip() != '':
                prot_seq = prot_seq + line.strip()
    
    if prot_id != '':
        seqs.append(prot_seq)
        ids.append(prot_id)
    return (ids,seqs)

def process_inputlabels(file):
    prot_id = ''
    prot_seq = ''
    labels = []
    ids = []
    for line in open(file, 'r'):
        if line[0] == '>':
            if prot_id != '':
                labels.append(prot_seq)
                ids.append(prot_id)
            
            prot_id = line.strip()
            prot_seq = ''
        elif line.strip() != '':
                prot_seq = prot_seq + line.strip()
    
    if prot_id != '':
        labels.append(prot_seq)
        ids.append(prot_id)
    return (ids,labels)

def convert_seq_ids(seq, tokenizer, max_len=200):
    # AACategoryLen = word_dim
    # probMatr = np.zeros((len(seq),AACategoryLen))
    # word_list = [i for i in k_mers(seq, kmer)]
    # word_list = sp.EncodeAsPieces(seq)
    tokens = tokenizer.tokenize(seq)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # padded_array = np.zeros(max_len, dtype=np.int32)
    # sq_shape = np.shape(token_ids)
    # print(sq_shape[0])
    # padded_array[:sq_shape[0]] = token_ids
    

    return (token_ids + [0] * (max_len - len(token_ids)))

# def pad(ids, max_seq_len):
#     x = []
#     for input_ids in ids:
#       input_ids = input_ids[:min(len(input_ids), max_seq_len - 2)]
#       input_ids = input_ids + [0] * (max_seq_len - len(input_ids))
#       x.append(np.array(input_ids))
#     return np.array(x)

def convertlabels_to_categorical(seq):
    
    letterDict = {} 
    letterDict["0"] = [1,0,0]
    letterDict["1"] = [0,1,0]
    letterDict["-"] = [0,0,1]
    AACategoryLen = 3 # 3 for '-'
    
    probMatr = np.zeros((len(seq),AACategoryLen))
    
    AANo  = 0
    for AA in seq:
        if not AA in letterDict:
           probMatr[AANo] = np.full(AACategoryLen, 0)
        elif AA=="-":
           probMatr[AANo]= letterDict[AA]
        else:
           probMatr[AANo]= [1-float(AA),float(AA),0]
        
        AANo += 1
    
    return probMatr

def initial_list(length):
    target=[None]*length
    for ind in range(len(target)):
        target[ind]=[]
    return target

def update_list(stride,num,comp,pred_score,pred):
    for j in range(comp):
        pred[j+num*stride].append(pred_score[j])  
    return pred

def convertlabels_to_binary(labels):
    return list(map(int, labels.replace('-', '0')))

def func_predict_scores_bilstm(testids, posscore, win, stride):
    regex = r"(\d+)left(\d+)"
    pred=""
    count = 1

    prediction_scores = {}
    for i in range(len(testids)):
        com=testids[i].split("_")
        com_len=len(com)
        seql=int(com[com_len-2])
        num_ind=com[com_len-1]  
        pred_score=posscore[i][:,1]  
        
        if re.search(regex,num_ind):
            match = re.search(regex, num_ind)
            num=int(match.group(1))
            comp=int(match.group(2))   
            if num==0:
                #pred=np.zeros(seql)
                pred=initial_list(seql)
                name="_".join(p for p in com[0:(com_len-1)])
                print(count, name)
                count+=1
            pred=update_list(stride,num,comp,pred_score,pred)
            #for j in range(comp):
                #if pred_score[j]>pred[j+num*stride]:
                    #pred[j+num*stride]=pred_score[j]   
            #score=' '.join([str(x) for x in pred])
            score=' '.join([str(np.median(x)) for x in pred])  
            prediction_scores[name] = score 
            print(score)
        elif num_ind=="0":
            #pred=np.zeros(seql)
            pred=initial_list(seql)
            num=int(num_ind)
            #for j in range(win):
                #if pred_score[j]>pred[j+num*stride]:
                    #pred[j+num*stride]=pred_score[j]
            pred=update_list(stride,num,win,pred_score,pred)
            name="_".join(p for p in com[0:(com_len-1)])
            print(count, name)
            count+=1    
            if (num*stride+win)==seql:
                #score=' '.join([str(x) for x in pred])
                score=' '.join([str(np.median(x)) for x in pred])
                prediction_scores[name] = score 
                print(score)
        else:    
            num=int(num_ind)
            #for j in range(win):
                #if pred_score[j]>pred[j+num*stride]:
                    #pred[j+num*stride]=pred_score[j]
            pred=update_list(stride,num,win,pred_score,pred)
            if (num*stride+win)==seql:
                #score=' '.join([str(x) for x in pred])
                score=' '.join([str(np.median(x)) for x in pred])
                prediction_scores[name] = score 
                print(score)
    return prediction_scores;
    
def round_score(x, threshold):
    if x > threshold:
        return 1
    else:
        return 0

def get_prediction_labels(scores_list, threshold):
    predict_labels = []
    for line in scores_list:
        arr = list(map(float, line.split(' ')))
        # arr =  list(map(round_score, arr))
        arr = [round_score(x, threshold) for x in arr]
        # arr = list(map(int, arr))
        arr = list(map(str, arr))
        predict_labels.append(''.join(arr))

    return predict_labels
    
def get_tp_tn_fp_fn(predict_vals, true_vals):
    tp, tn, fp, fn = 0, 0, 0, 0 
    for pval, tval in zip(predict_vals, true_vals):
        for p, t in zip(pval, tval):
            if p == '0' and t == '0':
                tn += 1
            elif p == '1' and t == '1':
                tp += 1
            elif p == '1' and t == '0':
                fp += 1
            elif p == '0' and t == '1':
                fn += 1
    
    return (tp, tn, fp, fn)
    
def get_results(tp, tn, fp, fn):
    print("Precision = ", (tp/(tp+fp)))
    print("Recall = ", (tp/(tp+fn)))
    print("Accuracy = ", ((tp+tn)/(tp+tn+fp+fn)))
    
def func_predict_scores_dense1(testids, posscore, win, stride):
    regex = r"(\d+)left(\d+)"
    pred=""
    count = 1

    prediction_scores = {}
    for i in range(len(testids)):
        com=testids[i].split("_")
        com_len=len(com)
        seql=int(com[com_len-2])
        num_ind=com[com_len-1]  
        # pred_score=posscore[i][:,1]  
        pred_score=posscore[i]

        # print("=> ", pred_score)
        
        if re.search(regex,num_ind):
            match = re.search(regex, num_ind)
            num=int(match.group(1))
            comp=int(match.group(2))   
            if num==0:
                #pred=np.zeros(seql)
                pred=initial_list(seql)
                name="_".join(p for p in com[0:(com_len-1)])
                print(count, name)
                count+=1
            pred=update_list(stride,num,comp,pred_score,pred)
            #for j in range(comp):
                #if pred_score[j]>pred[j+num*stride]:
                    #pred[j+num*stride]=pred_score[j]   
            #score=' '.join([str(x) for x in pred])
            score=' '.join([str(np.median(x)) for x in pred])  
            prediction_scores[name] = score 
            print(score)
        elif num_ind=="0":
            #pred=np.zeros(seql)
            pred=initial_list(seql)
            num=int(num_ind)
            #for j in range(win):
                #if pred_score[j]>pred[j+num*stride]:
                    #pred[j+num*stride]=pred_score[j]
            pred=update_list(stride,num,win,pred_score,pred)
            name="_".join(p for p in com[0:(com_len-1)])
            print(count, name)
            count+=1    
            if (num*stride+win)==seql:
                #score=' '.join([str(x) for x in pred])
                score=' '.join([str(np.median(x)) for x in pred])
                prediction_scores[name] = score 
                print(score)
        else:    
            num=int(num_ind)
            #for j in range(win):
                #if pred_score[j]>pred[j+num*stride]:
                    #pred[j+num*stride]=pred_score[j]
            pred=update_list(stride,num,win,pred_score,pred)
            if (num*stride+win)==seql:
                #score=' '.join([str(x) for x in pred])
                score=' '.join([str(np.median(x)) for x in pred])
                prediction_scores[name] = score 
                print(score)
    return prediction_scores;

def find_max_seq_len(data, tokenizer):
    max_seq_len = 0
    
    for row in tqdm(data, desc="Finding Max Seq Length", ncols="800px"):
        text, label = row[0], row[1]
        tokens = tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        max_seq_len = max(max_seq_len, len(token_ids))
    
    return max_seq_len

def convert_input_to_TapeBertPoolingOutput(sequence, tokenizer, model):
  max_sequence_length = len(sequence)
  seq_wo_dash = sequence[:len(sequence) if sequence.find("-") == -1 else sequence.find("-")]
  if max_sequence_length == len(seq_wo_dash):
    token_ids = torch.tensor([tokenizer.encode(seq_wo_dash[1:-1])])
  else:
    token_ids = torch.tensor([np.append(tokenizer.encode(seq_wo_dash), [0]*(max_sequence_length-len(seq_wo_dash)-2))], dtype=int)
  
  output = model(token_ids)
  # sequence_output = output[0]
  # pooled_output = output[1]

  return output[1][0].tolist()