from collections import defaultdict
from fire import Fire
import os
import json
import pandas as pd
import ir_datasets as irds
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
import pickle
import torch
import torch.nn as nn
import time

def dataset_from_idx(dataset, triplets=None, RND=42):
    frame = pd.DataFrame(dataset.docpairs_iter()) if not triplets else triplets
    docs = pd.DataFrame(dataset.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(dataset.queries_iter()).set_index('query_id').text.to_dict()

    frame['query'] = frame['query_id'].apply(lambda x: queries[x])
    frame['pid'] = frame['doc_id_a'].apply(lambda x: docs[x])
    frame['nid'] = frame['doc_id_b'].apply(lambda x: docs[x])

    return frame[['query', 'pid', 'nid']]

def iterate(df, style='triplet'):
    assert style in ['t5', 'triplet'], "Style must be either 't5' or 'triplet'"
    def t5_style():  
        OUTPUTS = ['true', 'false'] 
        while True:
            for row in df.itertuples():
                yield 'Query: ' + row.query + ' Document: ' + row.pid + ' Relevant:', OUTPUTS[0]
                yield 'Query: ' + row.query + ' Document: ' + row.nid + ' Relevant:', OUTPUTS[1]
    def triplet_style():
        while True:
            for row in df.itertuples():
                yield row.query, row.pid, row.nid

    return t5_style if style == 't5' else triplet_style

def swap_negatives(df : pd.DataFrame, lookup : dict):
    df['doc_id_b'] = df['qid'].apply(lambda x: lookup[x])
    return df

_logger = irds.log.easy()

def main(
        data_dir : str, # Stored docpairs subsample, currently CSV format
        dataset_name : str, # ir_datasets dataset name for text lookup
        out_dir : str, # Output directory for model and logs
        negative_dir : str = None, # Stored negative samples, currently serialized dict of form {qid : doc_id_b}
        train_steps : int = 1e6, # Number of training steps
        batch_size : int = 16, # Batch size 
        lr : float = 0.001): # Learning rate

    os.makedirs(out_dir, exist_ok=True)

    ## INIT DATA ##

    dataset = pd.read_csv(data_dir, index=False, header=0, names=['qid', 'doc_id_a', 'doc_id_b'])
    
    if negative_dir:
        with open(negative_dir, 'rb') as f:
            negatives = pickle.load(f)
        dataset = swap_negatives(dataset, negatives)

    corpus = irds.load(dataset_name)
    train = dataset_from_idx(dataset, corpus)

    logs = {
            'dataset': dataset,
            'model_name': 't5-base',
            'batch_size': batch_size,
            'lr': lr,
            'loss' : [],
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## INIT MODEL ##

    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ## TRAIN ##

    model.to(device)
    model.train()

    train_iter = iterate(train, style='t5')
    step = 0 
    total_loss = 0

    start = time.time()

    with _logger.pbar_raw(desc=f'train', total=train_steps) as pbar:
        while step < train_steps:
            for _ in range(batch_size):
                inp, out = [], []
                i, o = next(train_iter)
                inp.append(i)
                out.append(o)

            inp_ids = tokenizer(inp, return_tensors='pt', padding=True).input_ids.to(device)
            out_ids = tokenizer(out, return_tensors='pt', padding=True).input_ids.to(device)
    
            loss = model(input_ids=inp_ids, labels=out_ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            logs['loss'].append(loss.item())

            step += batch_size
            pbar.set_postfix({'loss': total_loss/step})
            pbar.update(batch_size)

    end = time.time() - start

    logs['time'] = end

    ## SAVE ##

    model.save_pretrained(os.path.join(out_dir, 'model'))
    with open(os.path.join(out_dir, 'logs.json'), 'w') as f:
        json.dump(logs, f)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)