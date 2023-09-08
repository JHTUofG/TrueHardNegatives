import os
import warnings
from fire import Fire
import pyterrier as pt
from tqdm.notebook import tqdm
from pyterrier_t5 import MonoT5ReRanker
from pyterrier.measures import RR, MAP, NDCG


os.environ['JAVA_HOME'] = "/usr/lib/jvm/jdk-17"
if not pt.started():
    pt.init()


def msmarco_generate(_dataset: str):
    """helper function
    """
    dataset = pt.get_dataset(_dataset)
    with pt.io.autoopen(dataset.get_corpus()[0], 'rt') as corpusfile:
        for l in corpusfile:
            docno, passage = l.split("\t")
            yield {'docno': docno, 'text': passage}


def main(
    output_name: str,
    limit: str = None,
    exp: str = 'exp1',
    corpus: str = r'msmarco_passage',
    eval_dataset: str = r'trec-deep-learning-passages',
    variant: str = 'test-2019',
    batch_size: int = 256,
    based: bool = False
):
    """main method for evaluation
    """

    def pipeline(bm25, index, model, batch_size):
        """helper function to build a pipeline"""
        return bm25 >> pt.text.get_text(index, "text") >> MonoT5ReRanker(model=model, batch_size=batch_size)
    
    with warnings.catch_warnings():
        # hide the useless warning output
        warnings.simplefilter('ignore', (FutureWarning, ))

        os.makedirs('results', exist_ok=True)

        bm25 = pt.BatchRetrieve.from_dataset(corpus, 'terrier_stemmed_text', wmodel='BM25', metadata=['docno', 'text'])
        dataset = pt.get_dataset(eval_dataset)

        # Try to load the saved Terrier index data structure from disk
        data_properties = './passage_index/data.properties'
        if os.path.isfile(data_properties):
            index = pt.IndexFactory.of(data_properties)

        # Build the Terrier index data structure if not found on disk
        else:
            iter_indexer = pt.IterDictIndexer("./passage_index")
            indexref = iter_indexer.index(msmarco_generate(eval_dataset), meta={'docno' : 20, 'text': 4096})
            index = pt.IndexFactory.of(indexref)
        
        if based:
            
            BASELINE_DIR = f'../Experiment_1/model_base/model_{batch_size}'
            NEGATIVE_DIR = f'../Experiment_1/model_new/model_{batch_size}'

            baseline_t5 = bm25 >> pt.text.get_text(index, "text") >> MonoT5ReRanker(model=BASELINE_DIR, batch_size=batch_size)
            retr_systems = [baseline_t5]
            names = ["Standard"]
            
            if exp != 'exp3_full':
                truenegative_t5 = bm25 >> pt.text.get_text(index, "text") >> MonoT5ReRanker(model=NEGATIVE_DIR, batch_size=batch_size)
                retr_systems.append(truenegative_t5)
                names.append("Hard")
        else:
            retr_systems = []
            names = []
            
        suffix = ''
        if limit:
            suffix = f'_limit_{limit}'

        if exp == 'exp2':
            # experiment setup for 'exp2'

            exp2_t5 = bm25 >> pt.text.get_text(index, "text") >> MonoT5ReRanker(model=f'model_exp2/model_{batch_size}_0.5', batch_size=batch_size)
            retr_systems.append(exp2_t5)
            names.append("E2 50%")

        elif exp == 'exp3':
            # experiment setup for 'exp3'

            fractions = [0.25, 0.5, 0.75]
            
            for f in fractions:
                retr_systems.append(pipeline(bm25, index, f'model_exp3/model_{batch_size}_{f}', batch_size))
                names.append(f"{str(f).split('.')[1]}%")
            
        elif exp == 'exp3_full':
            # experiment setup for 'exp3_full'

            fractions = [0.25, 0.5, 0.75, 1]
            exp3_t5_025 = pipeline(bm25, index, f'model_exp3/model_16_0.25', 16)
            retr_systems.append(exp3_t5_025)
            names.append("25%")
            
            for f in fractions:
                retr_systems.append(pipeline(bm25, index, f'model_exp3_full/model_{batch_size}_{f}{suffix}', batch_size))
                if f != 1:
                    names.append(f"{str(f).split('.')[1]}% 1.2m")
                else:
                    names.append("100% 1.2m")

        elif exp == 'exp4':
            # experiment setup for 'exp4'

            fractions = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

            for f in fractions:
                retr_systems.append(pipeline(bm25, index, f'model_exp4/model_{batch_size}_{f}{suffix}', batch_size))
                if f == 1:
                    names.append("100%")
                elif f == 0:
                    names.append("0%")
                else:
                    names.append(f"{str(f).split('.')[1]}0%")
        elif exp == 'exp4_extra':
            # experiment setup for 'exp4_extra'

            fractions = [0.25, 0.75]

            for f in fractions:
                retr_systems.append(pipeline(bm25, index, f'model_exp4/model_{batch_size}_{f}{suffix}', batch_size))
                names.append(f"{str(f).split('.')[1]}%")

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
