from unittest import TestCase

import torch

from models.utils import combine_mapping_networks, categorize_mappings
from models.networks.fc import FCGenerator


class UtilTests(TestCase):

    def setUp(self) -> None:
        self.state_dicts = [FCGenerator().state_dict() for _ in range(5)]
        self.mappings = [torch.eye(300, 300) for _ in range(2)]

    def test_combine_networks(self):
        child = combine_mapping_networks(*self.mappings, is_SO=True)
        self.assertTrue(
            torch.all(child['module.layer'] == self.mappings[0])
        )

    def test_combine_networks_r(self):
        for i in range(len(self.mappings)):
            self.mappings[i][0] = -self.mappings[i][0]
        child = combine_mapping_networks(*self.mappings, is_SO=False)
        self.assertTrue(
            torch.all(child['module.layer'] == self.mappings[0])
        )
