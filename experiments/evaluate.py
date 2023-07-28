from pyterrier.measures import RR, MAP, NDCG
from pyterrier_t5 import MonoT5ReRanker
from tqdm.notebook import tqdm
import pyterrier as pt
import pandas as pd
import argparse
import os
from fire import Fire
import warnings


os.environ['JAVA_HOME'] = "/usr/lib/jvm/jdk-17"
if not pt.started():
    pt.init()


def msmarco_generate():
    dataset = pt.get_dataset("trec-deep-learning-passages")
    with pt.io.autoopen(dataset.get_corpus()[0], 'rt') as corpusfile:
        for l in corpusfile:
            docno, passage = l.split("\t")
            yield {'docno' : docno, 'text' : passage}

def main(
    output_name : str,
    exp : str = 'exp1',
    corpus : str = r'msmarco_passage',
    eval_dataset : str = r'trec-deep-learning-passages',
    variant : str = 'test-2019',
    batch_size : int = 256
):
    
    def pipeline(bm25, index, model, batch_size):
        return bm25 >> pt.text.get_text(index, "text") >> MonoT5ReRanker(model=model, batch_size=batch_size)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', (FutureWarning, ))

        os.makedirs('results', exist_ok=True)

        bm25 = pt.BatchRetrieve.from_dataset(corpus, 'terrier_stemmed_text', wmodel='BM25', metadata=['docno', 'text'])
        dataset = pt.get_dataset(eval_dataset)

        ## Try to load the saved Terrier index data structure from disk
        data_properties = './passage_index/data.properties'
        if os.path.isfile(data_properties):
            index = pt.IndexFactory.of(data_properties)

        # Build the Terrier index data structure if not found on disk
        else:
            iter_indexer = pt.IterDictIndexer("./passage_index")
            indexref = iter_indexer.index(msmarco_generate(), meta={'docno' : 20, 'text': 4096})
            index = pt.IndexFactory.of(indexref)

        BASELINE_DIR = f'../Experiment_1/model_base/model_{batch_size}'
        NEGATIVE_DIR = f'../Experiment_1/model_new/model_{batch_size}'

        baseline_t5 = bm25 >> pt.text.get_text(index, "text") >> MonoT5ReRanker(model=BASELINE_DIR, batch_size=batch_size)
        truenegative_t5 = bm25 >> pt.text.get_text(index, "text") >> MonoT5ReRanker(model=NEGATIVE_DIR, batch_size=batch_size)

        retr_systems = [baseline_t5, truenegative_t5]
        names = ["Standard", "Hard"]

        if exp == 'exp2':

            exp2_t5 = bm25 >> pt.text.get_text(index, "text") >> MonoT5ReRanker(model=f'model_exp2/model_{batch_size}_0.5', batch_size=batch_size)
            retr_systems.append(exp2_t5)
            names.append("E2 50%")

        elif exp == 'exp3':

            # exp3_t5_025 = bm25 >> pt.text.get_text(index, "text") >> MonoT5ReRanker(model=f'model_exp3/model_{batch_size}_0.25', batch_size=batch_size)
            exp3_t5_025 = pipeline(bm25, index, f'model_exp3/model_{batch_size}_0.25', batch_size)
            retr_systems.append(exp3_t5_025)
            names.append("E3 25%")

            # exp2_t5 = bm25 >> pt.text.get_text(index, "text") >> MonoT5ReRanker(model=f'../Experiment_2/model_exp2/model_{batch_size}', batch_size=batch_size)
            exp2_t5 = pipeline(bm25, index, f'../Experiment_2/model_exp2/model_{batch_size}_0.5', batch_size)
            retr_systems.append(exp2_t5)
            names.append("E2 50%")

            # exp3_t5_075 = bm25 >> pt.text.get_text(index, "text") >> MonoT5ReRanker(model=f'model_exp3/model_{batch_size}_0.75', batch_size=batch_size)
            exp3_t5_075 = pipeline(bm25, index, f'model_exp3/model_{batch_size}_0.75', batch_size)
            retr_systems.append(exp3_t5_075)
            names.append("E3 75%")

        res = pt.Experiment(
            retr_systems,
            dataset.get_topics(variant=variant),
            dataset.get_qrels(variant=variant),
            eval_metrics=[RR(rel=2), MAP(rel=2), NDCG(cutoff=10), NDCG(cutoff=100)],
            baseline=0,
            names=names
        )

        res.to_csv(os.path.join('results', f'{variant}_{output_name}_{batch_size}.csv'))
    

if __name__ == '__main__':
    Fire(main)
