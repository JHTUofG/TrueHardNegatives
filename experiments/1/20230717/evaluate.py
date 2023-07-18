from pyterrier.measures import RR, MAP, NDCG
from pyterrier_t5 import MonoT5ReRanker
from tqdm.notebook import tqdm
import pyterrier as pt
import pandas as pd
import argparse
import os
from fire import Fire


os.environ['JAVA_HOME'] = "/usr/lib/jvm/jdk-17"
pt.init()


def msmarco_generate():
    dataset = pt.get_dataset("trec-deep-learning-passages")
    with pt.io.autoopen(dataset.get_corpus()[0], 'rt') as corpusfile:
        for l in corpusfile:
            docno, passage = l.split("\t")
            yield {'docno' : docno, 'text' : passage}

def main(
    output_name : str,
    corpus : str = r'msmarco_passage',
    eval_dataset : str = r'trec-deep-learning-passages',
    variant : str = 'test-2019',
    batch_size : int = 256,
):
    
    os.makedirs('results', exist_ok=True)
    
    bm25 = pt.BatchRetrieve.from_dataset(corpus, 'terrier_stemmed_text', wmodel='BM25', metadata=['docno', 'text'])
    dataset = pt.get_dataset(eval_dataset)
    
    iter_indexer = pt.IterDictIndexer("./passage_index")
    indexref = iter_indexer.index(msmarco_generate(), meta={'docno' : 20, 'text': 4096})
    index = pt.IndexFactory.of(indexref)
    
    BASELINE_DIR = r'model_base/model'
    NEGATIVE_DIR = r'model_new/model'
    
    baselinet5 = bm25 >> pt.text.get_text(index, "text") >> MonoT5ReRanker(model=BASELINE_DIR, batch_size=batch_size)
    truenegative_T5 = bm25 >> pt.text.get_text(index, "text") >> MonoT5ReRanker(model=NEGATIVE_DIR, batch_size=batch_size)
    
    res = pt.Experiment(
        [baselinet5, truenegative_T5],
        dataset.get_topics(variant=variant),
        dataset.get_qrels(variant=variant),
        eval_metrics=[RR(rel=2), MAP(rel=2), NDCG(cutoff=10)],
        baseline=0,
        names=["Standard MonoT5", "Hard Negative MonoT5"]
    )
    
    res.to_csv(os.path.join('results', f'{variant}_{output_name}.csv'))
    

if __name__ == '__main__':
    Fire(main)
