from unittest import TestCase
from collections import Counter

import numpy as np

from data.embedding_dataset import EmbeddingDataset


class EmbeddingTestCase(TestCase):

    def setUp(self) -> None:
        self.urls = EmbeddingDataset.download_urls
        self.url_names = list(EmbeddingDataset.download_urls.keys())

    def test_url_unique(self):
        c = Counter(self.urls.values())
        for url, occurrence in c.items():
            if occurrence != 1:
                print([k for k in self.urls.keys() if self.urls[k] == url])
        self.assertTrue(all([o == 1 for o in c.values()]))

    def test_name_matching_different(self):
        matched_urls = EmbeddingDataset.get_url_names('glove', 'skipgram', self.url_names)
        self.assertTupleEqual(tuple(matched_urls), ('0to0.1.glove.vec', '0to0.1.skipgram.vec'))

    def test_name_matching_self(self):
        matched_urls = EmbeddingDataset.get_url_names('glove', 'glove', self.url_names)
        self.assertTupleEqual(tuple(matched_urls), ('0to0.1.glove.vec', '0.1to0.2.glove.vec'))

    def test_name_matching_mock_target(self):
        matched_urls = EmbeddingDataset.get_url_names('glove', 'mock', self.url_names)
        self.assertTupleEqual(tuple(matched_urls), ('0to0.1.glove.vec', 'mock'))

    def test_create_mock_target(self):
        source_vecs = np.random.random([10000, 10]).astype('float32')
        target_vecs, mapping = EmbeddingDataset.create_mock_target(source_vecs)
        self.assertAlmostEqual(np.sum(mapping.T @ mapping - np.identity(10)), 0, places=5)
        self.assertTupleEqual(tuple(target_vecs.shape), (10000, 10))
        self.assertEqual(source_vecs.dtype, target_vecs.dtype)
