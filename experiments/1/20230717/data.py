import ir_datasets
import pandas as pd
from tqdm import tqdm
from utils import save_data


STORAGE_DIR = r'data'


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
            negative_lookup[qid] = tmp
            
    save_data(negative_lookup, STORAGE_DIR, "negative_lookup")
    
    return negative_lookup

def sample_df(df, n, save_as):
    new_df = df.sample(n=n)
    save_data(new_df, STORAGE_DIR, save_as)
    
    return new_df

def true_negatives(df, negative_lookup):
    new_df = df.copy()
    new_df['doc_id_b'] = new_df.apply(lambda x : negative_lookup[x.query_id][0].split('-')[0] if x.query_id in negative_lookup.keys() else x.doc_id_b, axis=1)
    save_data(new_df, STORAGE_DIR, 'truenegative_docpairs')
    
    return new_df

def dataset_from_idx(dataset, triplets, save_as):
    frame = triplets
    docs = pd.DataFrame(dataset.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(dataset.queries_iter()).set_index('query_id').text.to_dict()

    frame['query'] = frame['query_id'].apply(lambda x: queries[x])
    frame['pid'] = frame['doc_id_a'].apply(lambda x: docs[x])
    frame['nid'] = frame['doc_id_b'].apply(lambda x: docs[x])
    
    new_frame = frame[['query', 'pid', 'nid']]
    
    new_frame.to_csv(f'{STORAGE_DIR}/{save_as}.csv')

    return new_frame