import ir_datasets
import pandas as pd
from math import ceil
from tqdm import tqdm
from utils import save_data
from torch.utils.data import Dataset


STORAGE_DIR = r'data'
count = 0
OUTPUTS = ["true", "false"]

class NegativeDataset:
    # def __init__(self, pairs, bm25_idx, true_idx, corpus):
    def __init__(self, pairs, bm25_idx, true_idx, docs, queries):
        self.bm25_idx = bm25_idx
        self.true_idx = true_idx
        self.docs = docs
        queries = queries
        # self.docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
        # queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()

        self.data = [(queries[q], self.docs[p]) for q, p in pairs]

    def __len__(self):
        return len(self.data)
    
    def get_item(self, idx, hard=False):
        q, p = self.data[idx]
        n = self.true_idx[idx] if hard else self.bm25_idx[idx]
        
        return q, p, self.docs[n]

class NegativeLoader:
    def __init__(self, dataset, batch_size, initial_fraction=0.5) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.fraction = initial_fraction
    
    def __len__(self):
        return len(self.dataset)
    
    def format(self, q, d):
        return 'Query: ' + q + ' Document: ' + d + ' Relevant:'
    
    def get_batch(self, idx):
        num_true = ceil(self.fraction * self.batch_size)
        
        inp, label = [], []
        
        for i, j in enumerate(range(idx * self.batch_size, (idx + 1) * self.batch_size)):
            hardness = i < num_true
            
            q, p, n = self.dataset.get_item(j, hardness)
            
            inp.append(self.format(q, p))
            inp.append(self.format(q, n))
            label.append(OUTPUTS[0])
            label.append(OUTPUTS[1])
        
        return inp, label
    
class TestNegativeDataset(Dataset):
    def __init__(self, pairs, bm25_idx, true_idx, docs, queries, batch_size, initial_fraction=0.5):
        self.bm25_idx = bm25_idx
        self.true_idx = true_idx
        self.docs = docs
        queries = queries
        # self.dataset = dataset
        self.batch_size = batch_size
        self.fraction = initial_fraction

        self.data = [(queries[q], self.docs[p]) for q, p in pairs]

    def __len__(self):
        return int(len(self.data) / self.batch_size)
    
    def get_item(self, idx, hard=False):
        q, p = self.data[idx]
        n = self.true_idx[idx] if hard else self.bm25_idx[idx]
        
        return q, p, self.docs[n]
    
    def format(self, q, d):
        return 'Query: ' + q + ' Document: ' + d + ' Relevant:'
    
    def __getitem__(self, idx):
        # if idx < (len(self.data) / self.batch_size):
        num_true = ceil(self.fraction * self.batch_size)

        inp, label = [], []

        for i, j in enumerate(range(idx * self.batch_size, (idx + 1) * self.batch_size)):
            hardness = i < num_true

            q, p, n = self.get_item(j, hardness)

            inp.append(self.format(q, p))
            inp.append(self.format(q, n))
            label.append(OUTPUTS[0])
            label.append(OUTPUTS[1])
        
        return inp, label

def collect_negatives() -> dict:
    negatives = ir_datasets.load("msmarco-qna/train")
    qrels = pd.DataFrame(negatives.qrels_iter())
    qid_grouped = qrels.groupby('query_id')
    
    negative_lookup = {}

    for group in tqdm(qid_grouped):
        tmp = []
        zero_relevance = True

        for _, row in group[1].iterrows():
            if row.relevance == 1:
                zero_relevance = False
                break
            else:
                tmp.append(row.doc_id)
        if not zero_relevance and len(tmp) != 0:
            qid = group[0]
            negative_lookup[qid] = tmp[0].split('-')[0]
            
    save_data(negative_lookup, STORAGE_DIR, "negative_lookup")
    
    return negative_lookup

def sample_df(df, n, save_as) -> pd.DataFrame:
    new_df = df.sample(n=n)
    new_df.to_csv(f'{STORAGE_DIR}/{save_as}.csv', header=False, index=False)
    
    return new_df

def cross_sampling(docpairs : pd.DataFrame, negative_lookup : dict, n : int = 0, full : bool = False) -> pd.DataFrame:
    sampled = []
    neg_keys = negative_lookup.keys()
    
    for idx, row in tqdm(docpairs.iterrows(), total=docpairs.shape[0]):
        if len(sampled) < n or full:
            if row['query_id'] in neg_keys:
                sampled.append([row['query_id'], row['doc_id_a'], row['doc_id_b']])
        else:
            break
    sampled_df = pd.DataFrame(sampled)
    sampled_df.to_csv(f'{STORAGE_DIR}/filtered_docpairs.csv', header=False, index=False)
    
    return sampled_df
                

def true_negatives(df, negative_lookup) -> tuple[pd.DataFrame, int]:
    
    def counter(x):
        global count
        count += 1
        return x.doc_id_b
    
    new_df = df.copy()
    new_df['doc_id_b'] = new_df['query_id'].apply(lambda x : negative_lookup[x])
    ## new_df['doc_id_b'] = new_df.apply(lambda x : negative_lookup[x.query_id] if x.query_id in negative_lookup.keys() else counter(x), axis=1)
    # new_df['doc_id_b'] = new_df.apply(lambda x : negative_lookup[x.query_id][0].split('-')[0] if x.query_id in negative_lookup.keys() else count(x), axis=1)
    # ave_data(new_df, STORAGE_DIR, 'truenegative_docpairs')
    new_df.to_csv(f'{STORAGE_DIR}/truenegative_docpairs.csv', header=False, index=False)
    
    return new_df, count

def dataset_from_idx(dataset, triplets, save_as) -> pd.DataFrame:
    frame = triplets
    docs = pd.DataFrame(dataset.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(dataset.queries_iter()).set_index('query_id').text.to_dict()

    frame['query'] = frame['query_id'].apply(lambda x: queries[x])
    frame['pid'] = frame['doc_id_a'].apply(lambda x: docs[x])
    frame['nid'] = frame['doc_id_b'].apply(lambda x: docs[x])
    
    new_frame = frame[['query', 'pid', 'nid']]
    
    new_frame.to_csv(f'{STORAGE_DIR}/{save_as}.csv', header=False, index=False)

    return new_frame