from unittest import TestCase

import torch

from ..embedding_evaluator import EmbeddingEvaluator


class EmbeddingEvaluatorTest(TestCase):

    def test_l2_distance(self):
        a = torch.rand([123, 234])
        b = a
        distance = EmbeddingEvaluator.l2_distance(a, b)
        self.assertTrue(torch.all(distance.diagonal() == 0))

    def test_cosine_distance(self):
        a = torch.rand([123, 234])
        b = a
        distance = EmbeddingEvaluator.cosine_distance(a, b)
        self.assertTrue(torch.all(distance.diagonal() < 1e-5))
