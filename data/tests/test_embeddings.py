from unittest import TestCase
from collections import Counter

from data.embedding_dataset import EmbeddingDataset


class EmbeddingTestCase(TestCase):

    def setUp(self) -> None:
        self.urls = EmbeddingDataset.download_urls

    def test_url_unique(self):
        c = Counter(self.urls.values())
        for url, occurrence in c.items():
            if occurrence != 1:
                print([k for k in self.urls.keys() if self.urls[k] == url])
        self.assertTrue(all([o == 1 for o in c.values()]))
