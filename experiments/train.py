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
# from tqdm.notebook import tqdm
from tqdm import tqdm
from data import NegativeDataset, NegativeLoader, TestNegativeDataset
from utils import load_data
from torch.utils.data import DataLoader
from accelerate import Accelerator
import warnings


def iterate(df, style='triplet'):
    assert style in ['t5', 'triplet'], "Style must be either 't5' or 'triplet'"
    def t5_style(df):
        OUTPUTS = ['true', 'false']
        while True:
            for row in df.itertuples():
                yield 'Query: ' + row.query + ' Document: ' + row.pid + ' Relevant:', OUTPUTS[0]
                yield 'Query: ' + row.query + ' Document: ' + row.nid + ' Relevant:', OUTPUTS[1]
    def triplet_style():
        while True:
            for row in df.itertuples():
                yield row.query, row.pid, row.nid

    return t5_style(df) if style == 't5' else triplet_style


_logger = irds.log.easy()


def main(
        out_dir : str, # Output directory for model and logs
        dataset_name : str,
        cuda : str = '', # Switch cuda device with id
        fraction : float = 0.5,
        train_name : str = None,
        train_steps : int = 1e5, # Number of training steps
        batch_size : int = 16, # Batch size 
        lr : float = 0.001, # Learning rate
        mode : str = 'default' # default training mode, also have 'BM25-Warm-Up'
):
    
    if train_steps % batch_size != 0:
        return f"Error: train_steps {train_steps} not divisible by batch_size {batch_size}."
    
    def train_common(tokeniser, optimiser, inp, out, model, device, total_loss, logs, step):
    # def train_common(tokeniser, optimiser, model, device, total_loss, logs, step, inp_ids, out_ids):
        inp_ids = tokeniser(inp, return_tensors='pt', padding=True).input_ids.to(device)
        out_ids = tokeniser(out, return_tensors='pt', padding=True).input_ids.to(device)
        
        loss = model(input_ids=inp_ids, labels=out_ids).loss

        # loss.backward()
        accelerator.backward(loss)
        optimiser.step()
        optimiser.zero_grad()

        gathered_losses = [l for l in accelerator.gather(loss).tolist() if str(l) != 'nan']

        # total_loss += loss.item()
        # logs['loss'].append(loss.item())
        total_loss += sum(gathered_losses)
        logs['loss'].extend(gathered_losses)

        # step += (batch_size * len(gathered_losses))
        step = batch_size * len(gathered_losses)
        
        return total_loss, logs, step

    os.makedirs(out_dir, exist_ok=True)

    ## INIT DATA ##

    if train_name is not None and mode == 'default':
        dataset = pd.read_csv(f'data/{dataset_name}.csv', index_col=False, header=None, names=['query_id', 'doc_id_a', 'doc_id_b'])
        train = pd.read_csv(f'data/{train_name}.csv', index_col=False, header=None, names=['query', 'pid', 'nid'])
    else:
        dataset = irds.load(dataset_name)
        BM25_docpairs = pd.read_csv(f'../data/new_docpairs.csv', index_col=False, header=None, names=['query_id', 'doc_id_a', 'doc_id_b'])
        true_docpairs = pd.read_csv(f'../data/truenegative_docpairs.csv', index_col=False, header=None, names=['query_id', 'doc_id_a', 'doc_id_b'])

    logs = {
            'model_name': 't5-base',
            'batch_size': batch_size,
            'lr': lr,
            'loss' : [],
        }

    if cuda != '':
        cuda = f':{cuda}'

    # device = torch.device(f'cuda{cuda}' if torch.cuda.is_available() else 'cpu')
    accelerator = Accelerator()
    # device = ''
    device = accelerator.device

    ## INIT MODEL ##
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    # model = T5ForConditionalGeneration.from_pretrained('t5-base', device_map="auto")
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', (FutureWarning, ))
        tokeniser = T5Tokenizer.from_pretrained('t5-base')

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    ## TRAIN ##

    # model.to(device)
    model.train()

    step = 0
    total_loss = 0

    start = time.time()

    pbar = tqdm(total=train_steps)

    if mode == 'default':
        train_iter = iterate(train, style='t5')

        while step < train_steps:
            inp, out = [], []
            for _ in range(batch_size):
                i, o = next(train_iter)
                inp.append(i)
                out.append(o)

            total_loss, logs, step = train_common(tokeniser, optimiser, inp, out, model, device, total_loss, logs, step)

            pbar.set_postfix({'loss': total_loss/step})
            pbar.update(batch_size)

    elif mode == 'BM25-Warm-Up':

        pairs = BM25_docpairs[['query_id', 'doc_id_a']].values.astype(str).tolist()
        bm25_idx = BM25_docpairs['doc_id_b'].values.astype(str)
        true_idx = true_docpairs['doc_id_b'].values.astype(str)
        # neg_dataset = NegativeDataset(pairs, bm25_idx, true_idx, dataset)
        reduced_docs = load_data('data', 'reduced_docs')
        reduced_queries = load_data('data', 'reduced_queries')
        # neg_dataset = NegativeDataset(pairs, bm25_idx, true_idx, reduced_docs, reduced_queries)
        # train_iter = NegativeLoader(neg_dataset, batch_size, initial_fraction=fraction)
        test_dataset = TestNegativeDataset(pairs, bm25_idx, true_idx, reduced_docs, reduced_queries, batch_size, initial_fraction=fraction)
        test_dataloader = DataLoader(test_dataset, num_workers=8)

        model, optimiser, data = accelerator.prepare(model, optimiser, test_dataloader)
        
        for source, targets in data:
            source = list(sum(source, ()))
            targets = list(sum(targets, ()))
            total_loss, logs, _step = train_common(tokeniser, optimiser, source, targets, model, device, total_loss, logs, step)
            step += _step

            pbar.set_postfix({'loss': total_loss/step})
            pbar.update(_step)

#         for i in range(int(train_steps / batch_size)):
#             inp, out = train_iter.get_batch(i)

#             total_loss, logs, step = train_common(tokeniser, optimiser, inp, out, model, device, total_loss, logs, step)

#             pbar.set_postfix({'loss': total_loss/step})
#             pbar.update(batch_size)
#             # torch.cuda.empty_cache()

    end = time.time() - start

    logs['time'] = end

    ## SAVE ##
    out_folder = f'model_{batch_size}_{fraction}'
    model.module.save_pretrained(os.path.join(out_dir, out_folder))
    with open(os.path.join(out_dir, out_folder, 'logs.json'), 'w') as f:
        json.dump(logs, f)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)