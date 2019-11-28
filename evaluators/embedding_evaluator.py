# coding=utf-8

from functools import partial
import os
from typing import List, Dict

import torch
import numpy as np

from .base_evaluator import BaseEvaluator

MUSE_EVALUATION_DATA_PATH = './crosslingual/dictionaries/'


class EmbeddingEvaluator(BaseEvaluator):

    def __init__(self, opt, model, dataset, k=5, max_eval_size=1024):
        super().__init__(opt, model, dataset)
        self.k = k
        self.max_eval_size = max_eval_size
        self.source_word2idx = dataset.source_word2idx
        self.source_idx2word = dataset.source_idx2word
        self.target_word2idx = dataset.target_word2idx
        self.evaluation_size = opt.evaluation_size
        self.score_fns = {
            'in': self._get_previously_predicted_scores,
            'muse-nn-en': partial(self._get_muse_nn_scores, language='en'),
            'muse-csls-en':partial(self._get_muse_csls_scores, language='en')
        }
        self.score_name = opt.score_name[0]  # support only one score for now
        self.muse_source = None

    def get_current_scores(self):
        return self.score_fns[self.score_name]()

    def _get_previously_predicted_scores(self):
        """
        evaluate on the received training data directly
        """
        inp = self.model.inputs['source_idx']
        source_words = [self.source_idx2word[idx.item()] for idx in inp]
        predicted_embedding = self.model.get_output()
        predicted_embedding = predicted_embedding.cpu()
        target_idx, predicted_embedding = self._filter_mismatched_vocab(
            source_words, predicted_embedding,
        )

        target_embedding = self.dataset.target_vecs  # shape: (V, E)
        target_embedding = torch.from_numpy(target_embedding)
        distance = self.cosine_distance(predicted_embedding, target_embedding)  # shape: (N, V)
        top_k_distance, top_k_idx = torch.topk(distance, k=self.k, largest=False, dim=-1)
        # top_k_idx.shape: (N, k)

        precisions = {
            f'P@{k}': (top_k_idx[:, :k] == target_idx).float().sum(-1).mean().item()
            for k in range(1, self.k + 1)
        }

        mean_distance = top_k_distance.mean().item()
        mean_min_distance = top_k_distance[:, 0].mean().item()
        mean_max_distance = top_k_distance[:, -1].mean().item()
        return {
            **precisions,
            'mean_distance': mean_distance,
            'mean_min_distance': mean_min_distance,
            'mean_max_distance': mean_max_distance,
        }

    def _filter_mismatched_vocab(self, source_words: List[str], predicted_embedding: torch.Tensor):
        indices, target_idx = [], []
        for i, word in enumerate(source_words):
            if word in self.target_word2idx:
                indices.append(i)
                target_idx.append(self.target_word2idx[word])
        target_idx = np.array(target_idx, dtype=int)
        target_idx = torch.from_numpy(target_idx).to(device=predicted_embedding.device)
        target_idx = target_idx.unsqueeze(1)
        predicted_embedding = predicted_embedding.index_select(
            dim=0,
            index=torch.Tensor(indices).long(),
        )
        return target_idx, predicted_embedding

    def _get_muse_nn_scores(self, language='en'):
        if self.muse_source is None:
            self.muse_source = self._load_muse_dictionary(language)

        batch_size = self.opt.batch_size
        results = []
        source_vecs, source_idx = self.muse_source['vecs'], self.muse_source['idx']
        for i_batch in range(len(source_vecs) // batch_size):
            batch_data = source_vecs[i_batch * batch_size: (i_batch + 1) * batch_size]
            batch_idx = source_idx[i_batch * batch_size: (i_batch + 1) * batch_size]
            batch_data = torch.from_numpy(batch_data).to(self.model.device)
            batch_idx = torch.from_numpy(batch_idx).to(self.model.device)
            self.model.set_input({'source': batch_data, 'source_idx': batch_idx})
            self.model.forward()
            results.append(self._get_previously_predicted_scores())

        return self.aggregate_results(results)

    def _get_muse_csls_scores(self, language='en'):
        '''
        Modify from MUSE
        Compute cross-domain similarity local scaling of embeddings
        '''
        if self.muse_source is None:
            self.muse_source = self._load_muse_dictionary(language)

        # prepare evaluate data
        muse_source_words = [ self.source_idx2word[idx] for idx in self.muse_source['idx'] ]
        target_idx = [ self.target_word2idx[word] for word in muse_source_words if word in self.target_word2idx ]
        target_idx = torch.tensor(target_idx).long().to(self.model.device).unsqueeze(1)

        # get all source embedding after mapping
        batch_data = torch.from_numpy(self.dataset.source_vecs).to(self.model.device)
        batch_idx = torch.tensor(list(self.source_idx2word.keys())).long().to(self.model.device)
        self.model.set_input({'source': batch_data, 'source_idx': batch_idx})
        self.model.forward()

        # normalize word embeddings
        emb1 = self.model.get_output().data
        emb2 = torch.from_numpy(self.dataset.target_vecs).to(self.model.device)
        emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
        emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

        # get average k-nearest-neighbors distance between all embedding pairs
        average_dist1 = self.get_nn_avg_dist(emb2, emb1, self.k, self.opt.batch_size)
        average_dist2 = self.get_nn_avg_dist(emb1, emb2, self.k, self.opt.batch_size)

        # queries -> scores
        query = emb1[self.muse_source['idx']]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[self.muse_source['idx']][:, np.newaxis])
        scores.sub_(average_dist2[np.newaxis, :])

        top_k_distance, top_k_idx = torch.topk(scores, k=self.k, largest=True, dim=-1)

        precisions = {
            f'P@{k}': (top_k_idx[:, :k] == target_idx).float().sum(-1).mean().item()
            for k in range(1, self.k + 1)
        }

        mean_distance = top_k_distance.mean().item()
        mean_min_distance = top_k_distance[:, 0].mean().item()
        mean_max_distance = top_k_distance[:, -1].mean().item()
        return {
            **precisions,
            'mean_distance': mean_distance,
            'mean_min_distance': mean_min_distance,
            'mean_max_distance': mean_max_distance,
        }

    def _load_muse_dictionary(self, language):
        dictionary_path = os.path.join(
            MUSE_EVALUATION_DATA_PATH,
            f'{language}-{language}.5000-6500.txt',
        )
        try:
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                pair_words = [line.split()[:2] for line in f]
        except FileNotFoundError:
            print('Please download evaluation data with ./embedding/get_evaluation!')
            exit()
        pair_words = [
            pair for pair in pair_words
            if pair[0] in self.source_word2idx and pair[1] in self.target_word2idx
        ]
        source_idx = np.array([self.source_word2idx[w] for w, _ in pair_words])
        source_vecs = self.dataset.source_vecs[source_idx]
        print(f'Evaluating on {len(source_vecs)} words from {dictionary_path}.')
        return {'vecs': source_vecs, 'idx': source_idx}

    @staticmethod
    def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # need better implementation
        distance = torch.zeros([a.shape[0], b.shape[0]], device=a.device)
        for i, x in enumerate(a):
            x = x.view(1, -1)
            distance[i] = ((b - x) ** 2).sum(dim=-1).sqrt()
        return distance

    @staticmethod
    def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a / a.norm(2, dim=-1, keepdim=True)
        b = b / b.norm(2, dim=-1, keepdim=True)
        inner_product = torch.einsum('ab,cb->ac', [a, b])
        return 1 - inner_product

    @staticmethod
    def aggregate_results(results: List[Dict]):
        return {
            key: np.mean([r[key] for r in results])
            for key in results[0].keys()
        }

    @staticmethod
    def get_nn_avg_dist(emb, query, knn, batchsize):
        """
        Modify from MUSE, may add Faiss in the future
        Compute the average distance of the `knn` nearest neighbors
        for a given set of embeddings and queries.
        Use Faiss if available.
        """
        bs = batchsize
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1))
        all_distances = torch.cat(all_distances)
        return all_distances
