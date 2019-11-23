import torch
import numpy as np

from .base_evaluator import BaseEvaluator


class EmbeddingEvaluator(BaseEvaluator):

    def __init__(self, opt, model, dataset, k=1):
        super().__init__(opt, model, dataset)
        self.model = model
        self.k = k
        self.source_idx2word = dataset.source_idx2word
        self.target_word2idx = dataset.target_word2idx
        self.evaluation_size = opt.evaluation_size

    def get_current_scores(self):
        inp = self.model.inputs['source_idx']
        source_words = [self.source_idx2word[idx.item()] for idx in inp]
        target_idx = np.array(
            [self.target_word2idx[word] for word in source_words],
            dtype=int,
        )
        predicted_embedding = self.model.get_output().cpu()
        target_embedding = self.dataset.target_vecs  # shape: (V, E)
        target_embedding = torch.from_numpy(target_embedding)
        distance = self.cosine_distance(predicted_embedding, target_embedding)  # shape: (N, V)
        top_k_distance, top_k_idx = torch.topk(distance, k=self.k, largest=False, dim=-1)
        # top_k_idx.shape: (N, k)

        target_idx = torch.from_numpy(target_idx).to(device=predicted_embedding.device)
        target_idx = target_idx.unsqueeze(1)
        precision = (top_k_idx == target_idx).float().mean()

        mean_distance = top_k_distance.mean()
        return {f'precision@{self.k}': precision.item(), 'mean_distance': mean_distance.item()}

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
