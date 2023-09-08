import ir_datasets
import pandas as pd
from math import ceil
from tqdm import tqdm
from utils import save_data
from torch.utils.data import Dataset


STORAGE_DIR = r'data'
count = 0
OUTPUTS = ["true", "false"]


class NegativeDataset(Dataset):
    """
    class that used to mix the true hard negative into the dataset
    """
    def __init__(self, pairs, bm25_idx, true_idx, corpus, batch_size, initial_fraction=0.5):
        self.bm25_idx = bm25_idx
        self.true_idx = true_idx
        self.docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
        queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()

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
        # mix a certain ratio of true hard negatives
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
    """
    function used to build the negative lookup
    :return: dict - a dictionary of negative lookup
    """
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


def sample_df(df: pd.DataFrame, n: int, save_as: str) -> pd.DataFrame:
    """
    sampling a certain amount of data from a DataFrame
    :param df: pd.DataFrame - the candidate DataFrame
    :param n: int - the size of the new DataFrame
    :param save_as: str - name of the file
    :return: pd.DataFrame - a new DataFrame
    """
    new_df = df.sample(n=n)
    new_df.to_csv(f'{STORAGE_DIR}/{save_as}.csv', header=False, index=False)
    
    return new_df


def cross_sampling(docpairs: pd.DataFrame, negative_lookup: dict, n: int = 0, full: bool = False) -> pd.DataFrame:
    """
    cross sampling the docpairs with the negative lookup to generate the intersection subset
    :param docpairs: pd.DataFrame
    :param negative_lookup: dict
    :param n: int - cross-sample a certain amount of data
    :param full: bool - cross-sample the full size data if True
    :return: pd.DataFrame - a cross-sampled new DataFrame
    """
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
                

def true_negatives(df: pd.DataFrame, negative_lookup: dict, prefix: str = '') -> pd.DataFrame:
    """
    using the negative lookup to build the true negatives
    :param df: pd.DataFrame - the docpairs
    :param negative_lookup: dict - the pre-built negative lookup
    :param prefix: str - a prefix for the file name
    :return: pd.DataFrame - a new DataFrame with true negatives
    """
    new_df = df.copy()
    new_df['doc_id_b'] = new_df['query_id'].apply(lambda x: negative_lookup[x])
    new_df.to_csv(f'{STORAGE_DIR}/{prefix}_truenegative_docpairs.csv', header=False, index=False)
    
    return new_df


def dataset_from_idx(dataset: ir_datasets.datasets.base.Dataset, triplets: pd.DataFrame, save_as: str) -> pd.DataFrame:
    """
    default method for building the dataset

    """
    frame = triplets
    docs = pd.DataFrame(dataset.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(dataset.queries_iter()).set_index('query_id').text.to_dict()

    frame['query'] = frame['query_id'].apply(lambda x: queries[x])
    frame['pid'] = frame['doc_id_a'].apply(lambda x: docs[x])
    frame['nid'] = frame['doc_id_b'].apply(lambda x: docs[x])
    
    new_frame = frame[['query', 'pid', 'nid']]
    
    new_frame.to_csv(f'{STORAGE_DIR}/{save_as}.csv', header=False, index=False)

    return new_frame