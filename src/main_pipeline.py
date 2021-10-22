import os
import string
import time
from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 200)

import pymorphy2
from scipy.stats import gmean, hmean
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS  # noqa
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import silhouette_score, adjusted_rand_score

from hdbscan import HDBSCAN  # noqa

from substs_loading import load_substs


class Lemmatizer:
    def __init__(self, mode='default'):
        self.mode = mode
        self.morph = pymorphy2.MorphAnalyzer()

    def lemmatize_row_simple(self, probs_row: List[Tuple[float, str]]):
        lemmatizer = lambda x: self.morph.parse(x.strip())[0].normal_form

        total_prob = defaultdict(int)
        for prob, word in probs_row:
            total_prob[lemmatizer(word)] += prob
        norm_coef = sum(prob for word, prob in total_prob.items())
        new_prob_list = sorted([(prob / norm_coef, word) for word, prob in total_prob.items()], reverse=True)
        return new_prob_list

    def lemmatize_row_averaged(self, probs_row: List[Tuple[int, str]]):
        raise NotImplementedError()

    def lemmatize_row(self, probs_row: List[Tuple[int, str]]):
        prepared_row = self.prepare_row(probs_row)
        if self.mode == 'simple':
            return self.lemmatize_row_simple(prepared_row)
        elif self.mode == 'averaged':
            return self.lemmatize_row_averaged(prepared_row)
        elif self.mode == 'default':
            return prepared_row

    def prepare_row(self, probs_row: List[Tuple[int, str]]):
        remove_punc = lambda word: word.translate(str.maketrans('', '', string.punctuation))
        prep_row = [(prob, remove_punc(word)) for prob, word in probs_row]

        split_nwords_substs = lambda row: list(chain(
            *[
                [(prob, spl_word) for spl_word in subst.split()]
                for prob, subst in row
            ]
        ))
        prep_row = split_nwords_substs(prep_row)

        return prep_row


class Vectorizer:
    """
        Take all texts with probs from all documents and vectorize is with some method.
    """

    def __init__(self, vec_mode='count',
                 comb_mode='harm', ood_mode='eps', ood_prob=1e-7, top_k=150,
                 feature_select_mode=None, keep_imp_proc=0.5,
                 **vectorizer_conf):
        """
            args:
                vec_mode:           type of Vectorizer from sklearn (or mb custom)
                with_probs:         multiply vectorizer numbers by prob. of subst
                comb_mode:          mode of combining probs from files
                ood_prob:           probability for words appearing only in one file
                vectorizer_conf:    settings for vectorizer

        """
        self.important_words = None
        if vec_mode == 'count':
            self.vectorizer = CountVectorizer(
                preprocessor=lambda x: x,
                **vectorizer_conf
            )
        elif vec_mode == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                tokenizer=lambda x: x,
                **vectorizer_conf
            )
        self.comb_mode = comb_mode
        self.ood_mode = ood_mode
        self.ood_prob = ood_prob
        self.top_k = top_k
        self.comb_mapping = {
            'geom': gmean,
            'harm': hmean,
            'mean': np.mean,
            'prod': np.prod,
        }
        self.feature_selection_mode = feature_select_mode
        self.keep_important_proc = keep_imp_proc

    def comb_prob(self, probs: Tuple[float]):
        return self.comb_mapping[self.comb_mode](probs)

    def unite_n_rows(self, probs_rows_df: Tuple[List[Tuple[float, str]]]):
        """
            Unite n rows from different files into one row and leaves only top_k substs.
        """
        probs_rows = tuple(probs_rows_df)
        XML_DICT_SIZE = 2.5 * 10 ** 6

        if self.ood_mode == 'res_mean':
            ood_probs = [(1 - sum(prob for prob, word in row)) / (XML_DICT_SIZE - len(row)) for row in
                         probs_rows]  # probs if subst from other file not presented
        else:
            ood_probs = [self.ood_prob for row in probs_rows]

        all_substs = set(word for prob, word in chain(*probs_rows) if word != '')
        file_word_probs = [
            {file_row_word: file_row_prob for file_row_prob, file_row_word in p_s_row}
            for p_s_row in probs_rows
        ]
        substs_probs = {
            word: tuple(file_probs.get(word, ood_probs[ind]) for ind, file_probs in enumerate(file_word_probs))
            for word in all_substs
        }
        new_probs_substs = sorted([(self.comb_prob(w_probs), word) for word, w_probs in substs_probs.items()],
                                  reverse=True)
        return new_probs_substs[:self.top_k]

    def unite_words_dfs(self, splitted_word_dfs: List[List[pd.DataFrame]]) -> List[pd.DataFrame]:
        """
        Structure of input data:
            List by files
                List by unique word
                    Df with substs
        """
        words_substs_probs = []
        for files_word_dfs in zip(*splitted_word_dfs):
            # Merge many dfs into one, to perform apply later
            res: pd.DataFrame = files_word_dfs[0]
            for ind, df in enumerate(files_word_dfs[1:]):
                res = pd.merge(res, df, left_index=True, right_index=True, suffixes=(None, f'{ind + 1}'))
            res = res.drop(columns=list(res.filter(regex='word\d+')))

            # Merging probs from different files
            new_probs = res.filter(regex='probs').apply(self.unite_n_rows, axis=1)
            res = res.drop(columns=list(res.filter(regex='substs_probs')))
            res['substs_probs'] = new_probs

            words_substs_probs.append(res)

        return words_substs_probs

    def vectorize_word_df(self, word_df: pd.DataFrame) -> np.ndarray:
        if self.important_words is not None:
            text_from_row = lambda row: ' '.join([word for prob, word in row if word in self.important_words])
        else:
            text_from_row = lambda row: ' '.join([word for prob, word in row])
        df_texts = list(word_df.substs_probs.apply(text_from_row))
        # print(df_texts[0])
        vectorized_texts = self.vectorizer.fit_transform(df_texts).toarray()
        return vectorized_texts

    def feature_selection(self, word_dfs_lists: List[pd.DataFrame], keep_proc=0.5):
        """
            Check chi2/mutual_inf for every subst in dataset and keeps set of some percent of most important words
        """
        all_words_df = pd.concat(word_dfs_lists)
        all_words_vectorized = self.vectorize_word_df(all_words_df)
        all_words_labels = np.concatenate([np.repeat(ind, df.shape[0]) for ind, df in enumerate(word_dfs_lists)])

        print("FEATURE SELECTION makes brrrr")
        ind2word = {ind: word for word, ind in self.vectorizer.vocabulary_.items()}
        num_remove = int(len(ind2word) * (1 - keep_proc))
        if self.feature_selection_mode == 'chi2':
            feature_imp = chi2(all_words_vectorized, all_words_labels)[0]
        else:
            feature_imp = mutual_info_classif(all_words_vectorized, all_words_labels, discrete_features=True)
        imp_order = np.argsort(feature_imp)
        self.important_words = {ind2word[ind] for ind in imp_order[num_remove:]}
        print("SELECTION FINISHED")

    def transform(self, file_word_dfs_list: List[List[pd.DataFrame]]) -> List[np.ndarray]:
        print("VECTORIZER TRANSFORM STARTED: ")
        st = time.time()
        words_dfs_list = self.unite_words_dfs(file_word_dfs_list)
        if self.feature_selection_mode is not None:
            self.feature_selection(words_dfs_list)
        words_vec_texts = [
            self.vectorize_word_df(df) for df in words_dfs_list
        ]
        print(f"Vectorizing finished in {time.time() - st} sec.")
        return words_vec_texts


class Clusterizer:
    def __init__(self, comp_mode='train', clust_mode='agg', cluster_range: Tuple[int, int] = (2, 6),
                 **clusterizer_config):
        self.clustering_mapping = {
            'agg': AgglomerativeClustering,
            'dbscan': DBSCAN,
            'hdbscan': HDBSCAN,
            'optics': OPTICS,
        }
        self.clusterizer = self.clustering_mapping[clust_mode](**clusterizer_config)
        self.cluster_range = range(*cluster_range)
        self.mode = comp_mode

    def _compute_all_labels(self, vectorized_texts: List[np.ndarray]):
        """
        Compute labels for all n_cluster in cluster_range
        """
        print("COMPUTING LABELS STARTED")
        self.all_pred_labels: List[Dict[int, np.ndarray]] = []
        for vec_texts in vectorized_texts:
            word_pred_labels = dict()
            for num_clust in self.cluster_range:
                self.clusterizer.set_params(n_clusters=num_clust)
                pred_labels = self.clusterizer.fit_predict(vec_texts)
                word_pred_labels[num_clust] = pred_labels
            self.all_pred_labels.append(word_pred_labels)
        print("COMPUTING LABELS FINISHING")

    def compute_metrics(self, vectorized_texts: List[np.ndarray] = None,
                        true_labels: List[np.ndarray] = None):
        self._compute_all_labels(vectorized_texts)
        self.all_slih_results = [
            {
                nc: silhouette_score(vec_word_text, pred_lab)
                for nc, pred_lab in self.all_pred_labels[ind_w].items()
            }
            for ind_w, vec_word_text in enumerate(vectorized_texts)
        ]
        if self.mode == 'train':
            self.all_ari_results = [
                {
                    nc: adjusted_rand_score(true_labels[ind_w], pred_lab)
                    for nc, pred_lab in all_word_pred_labels.items()
                }
                for ind_w, all_word_pred_labels in enumerate(self.all_pred_labels)
            ]

    def silhouette_metric(self):
        max_silh_clust = [
            max(word_silh_result.items(), key=lambda x: x[1]) for word_silh_result in self.all_slih_results
        ]
        if self.mode == 'test':
            return max_silh_clust
        max_silh_ari = [
            (nc, self.all_ari_results[w_ind][nc]) for w_ind, (nc, silh) in enumerate(max_silh_clust)
        ]
        return max_silh_ari

    def ari_metric(self):
        max_ari_results = [
            max(word_ari_result.items(), key=lambda x: x[1]) for word_ari_result in self.all_ari_results
        ]
        return max_ari_results

    def fixed_nc_ari_metric(self):
        fnc_ari_results = {
            nc: np.mean([all_word_ari[nc] for w_ind, all_word_ari in enumerate(self.all_ari_results)])
            for nc in self.cluster_range
        }
        return fnc_ari_results

    def get_best_metrics(self):
        print("best metrics")
        return {
            'silh': np.mean([silh for nc, silh in self.silhouette_metric()]),
            'ari': np.mean([ari for nc, ari in self.ari_metric()]),
            'fixed_nc_ari': max(self.fixed_nc_ari_metric().items(), key=lambda x: x[1])
        }

    def clusterize(self):
        silh_res = self.silhouette_metric()
        print(silh_res)
        pred_labels = [
            self.all_pred_labels[w_ind][nc] for w_ind, (nc, silh) in enumerate(silh_res)
        ]
        return pred_labels


class WSIPipeline:
    morph = pymorphy2.MorphAnalyzer()

    def __init__(self, mode='train'):
        self.lemmatizer: Optional[Lemmatizer] = None
        self.vectorizer: Optional[Vectorizer] = None
        self.clusterizer: Optional[Clusterizer] = None
        self.mode = mode

        self.files_dfs = []
        self.word2ind = None
        self.ind2word = None
        self.true_labels = None

    def load_files(self, data_directory: str, files: List[str]):
        self.files_dfs: List[pd.DataFrame] = []
        print(data_directory, files)
        for filename in files:
            full_path = os.path.realpath(data_directory + '/' + filename)
            if filename.endswith('.pkl'):
                print(full_path)
                df = pd.read_pickle(full_path)
            else:
                df = load_substs(full_path)
            tmp_df = df.filter(['index', 'context_id', 'word', 'substs_probs', 'gold_sense_id'])
            self.files_dfs.append(tmp_df)

        self.ind2word = dict(enumerate(self.files_dfs[0].word.unique()))
        self.word2ind = {word: ind for ind, word in self.ind2word.items()}

        if self.mode != 'test':
            self.true_labels = [
                self.files_dfs[0][self.files_dfs[0].word == uniq_w].gold_sense_id.to_numpy()
                for uniq_w in self.word2ind
            ]

    def lemmatize_probs_dfs(self, mode='default'):
        """
            Lemmatize all probs from dataset df
        """
        st = time.time()
        self.lemmatizer = Lemmatizer(mode=mode)
        tmp_files_dfs = []
        for ind, df in enumerate(self.files_dfs):
            tmp_df = df.copy()
            tmp_df.substs_probs = tmp_df.substs_probs.apply(self.lemmatizer.lemmatize_row)
            tmp_files_dfs.append(tmp_df)

        self.files_dfs = tmp_files_dfs
        print(f"Lemmatize finished in {time.time() - st} sec.")

    def vectorize_probs(self, **vectorizer_params):

        files_words_dfs = [
            [df[df.word == uniq_w] for uniq_w in self.word2ind]
            for df in self.files_dfs
        ]
        self.vectorizer = Vectorizer(**vectorizer_params)
        self.vectorized_words_substs: List[np.ndarray] = self.vectorizer.transform(files_words_dfs)

    def clusterize(self, **clustering_config):
        self.clusterizer = Clusterizer(comp_mode=self.mode, **clustering_config)
        self.clusterizer.compute_metrics(
            vectorized_texts=self.vectorized_words_substs,
            true_labels=self.true_labels
        )
        return self.clusterizer.clusterize()

    def get_cluster_metrics(self, **clustering_config):
        self.clusterizer = Clusterizer(**clustering_config)
        self.clusterizer.compute_metrics(
            vectorized_texts=self.vectorized_words_substs,
            true_labels=self.true_labels
        )
        return self.clusterizer.get_best_metrics()

    def write_result(self, preds_path: str, word_labels: List[np.array]):
        print(f"Writing result to file {preds_path}")
        result_df = self.files_dfs[0].copy().drop(columns='substs_probs')
        print(list(result_df))

        for word, ind in self.word2ind.items():
            mask = (result_df.word == word)
            # print(word, result_df.loc[mask, 'predict_sense_id'].shape)
            result_df.loc[mask, 'predict_sense_id'] = word_labels[ind]
            result_df.loc[mask, 'predict_sense_id'] = result_df.loc[mask, 'predict_sense_id'].astype(int)

        result_df.to_csv(preds_path, sep='\t', index=False)


# data_directory = "./subst_test/N_subwords"
# data_directory = "../subst_train/active_subst"
# filenames = [
#     # r"m(T)-lemmatized-simple.pkl",
#     # r"T(m)-lemmatized-simple.pkl",
#     # r"mm(T)-lemmatized-simple.pkl",
#     # r"T(mm)-lemmatized-simple.pkl",
#     r"mmm(T)-lemmatized-simple.pkl",
#     r"T(mmm)-lemmatized-simple.pkl",
#     r"T-and-mmm-lemmatized-simple.pkl",
#     r"mmm-and-T-lemmatized-simple.pkl",
# ]
default_wsi_conf = {
    'data_directory': r'../subst_test/active_subst',
    'files': [
        r"mmm(T)-lemmatized-simple.pkl",
        r"T(mmm)-lemmatized-simple.pkl",
        r"T-and-mmm-lemmatized-simple.pkl",
        r"mmm-and-T-lemmatized-simple.pkl",
        # r"mmm-or-T-lemmatized-simple.pkl",
        # r"T-or-mmm-lemmatized-simple.pkl",
        
    ],
}
default_lemma_conf = {
    'mode': 'default',
}
default_vec_conf = {
    'vec_mode': 'count',
    'comb_mode': 'mean',
    'ood_mode': 'eps',
    'top_k': 150,
    'ood_prob': 1e-7,
    'min_df': 2,
}
default_clustering_conf = {
    'clust_mode': 'agg',  # type of clustering model (agg, dbscan, optics)
    'affinity': 'cosine',  # affinity for clustering model
    'linkage': 'average',  # linkage for agg model
    'memory': './cache',  # memory for agg model
}

preds_dir = '../preds'
preds_filename = 'also+and+1e-7.tsv'
preds_full_path = os.path.realpath(preds_dir + '/' + preds_filename)

if __name__ == "__main__":
    wsi = WSIPipeline(mode='test')
    wsi.load_files(**default_wsi_conf)
    wsi.lemmatize_probs_dfs(**default_lemma_conf)
    wsi.vectorize_probs(**default_vec_conf)
    preds_labels = wsi.clusterize(**default_clustering_conf)
    wsi.write_result(preds_path=preds_full_path, word_labels=preds_labels)
