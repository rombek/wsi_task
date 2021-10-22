import sys
import time
from itertools import chain

import pandas as pd

from main_pipeline import Lemmatizer
from substs_loading import load_substs


def create_lemmatize_df(df: pd.DataFrame, mode='simple') -> pd.DataFrame:
    st = time.time()
    lemmatizer = Lemmatizer(mode=mode)
    tmp_df = df.filter(['index', 'context_id', 'word', 'substs_probs', 'gold_sense_id'])
    tmp_df.substs_probs = tmp_df.substs_probs.apply(lemmatizer.lemmatize_row)
    print(f"Lemmatize finished in {time.time() - st} sec.")
    return tmp_df


if __name__ == '__main__':
    """
        Usage: python lemmatize_df.py <file with substs> <output_name>
    """
    print(sys.argv)
    lemma_mode = 'simple'
    file_name = sys.argv[1]
    try:
        out_name = sys.argv[2]
    except IndexError:
        parts = file_name.split('.npz')
        print(parts)
        out_name = parts[0] + f'-lemmatized-{lemma_mode}'

    df = load_substs(file_name)
    lemma_df = create_lemmatize_df(df, mode=lemma_mode)
    lemma_df.to_pickle(f'{out_name}.pkl')
