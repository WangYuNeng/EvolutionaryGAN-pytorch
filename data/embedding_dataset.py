import os
import subprocess
import random

import numpy as np

from data.base_dataset import BaseDataset


class EmbeddingDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""

    download_urls = {
        '0to0.1.cbow.vec': 'https://drive.google.com/open?id=1-C3Nq9nXBzBsPI570IXkl81UlcBE-JYn',
        '0to0.1.skipgram.vec': 'https://drive.google.com/open?id=1-SPp0fIoGhHtbV4Z73bHrKIje-hUzD2D',
    }

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--download_root', type=str, default='./datasets/embedding',
                            help='root directory of dataset exist or will be saved')
        parser.add_argument('--source_dataset_name', type=str, default='cbow',
                            help='name of imported dataset, options: [cbow, fasttext]')
        parser.add_argument('--target_dataset_name', type=str, default='cbow',
                            help='name of imported dataset, options: [cbow, fasttext]')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.data_root = opt.download_root
        if not os.path.isdir(self.data_root):
            os.mkdir(self.data_root)

        self.source_name = opt.source_dataset_name
        self.target_name = opt.target_dataset_name
        self.source_url_name, self.target_url_name = self.get_url_names(
            self.source_name,
            self.target_name,
            list(self.download_urls.keys()),
        )
        for url_name in [self.source_url_name, self.target_url_name]:
            self.download_embeddings(url_name)

        self.source_vecs, self.source_word2idx, self.source_idx2word = self.load_embeddings(self.source_url_name)
        self.target_vecs, self.target_word2idx, self.target_idx2word = self.load_embeddings(self.target_url_name)

    @staticmethod
    def get_url_names(source_name: str, target_name: str, url_names: list):
        matched_url_names = []
        for match_name in [source_name, target_name]:
            for name in url_names:
                if match_name in name:
                    matched_url_names.append(name)
                    url_names.remove(name)
                    break
        return matched_url_names

    def download_embeddings(self, url_name: str):
        if os.path.exists(os.path.join(self.data_root, url_name)):
            return
        subprocess.run(
            [
                'perl',
                'scripts/download_drive.pl',
                self.download_urls[url_name],
                os.path.join(self.data_root, url_name),
            ]
        )

    def load_embeddings(self, url_name: str):
        file_path = os.path.join(self.data_root, url_name)
        words = []
        vecs = []
        with open(file_path, 'r') as f:
            for line in f:
                fields = line.strip().split()
                words.append(fields[0])
                vecs.append([float(v) for v in fields[1:]])
        vecs = np.asarray(vecs)
        word2idx = {w: i for i, w in enumerate(words)}
        idx2word = {i: w for i, w in enumerate(words)}
        return vecs, word2idx, idx2word

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        """
        target_index = (index + np.random.random(0, self.__len__())) % self.__len__()
        return {'source': self.source_vecs[index], 'target': self.target_vecs[target_index]}

    def __len__(self):
        """Return the total number of word-vectors."""
        if len(self.source_vecs) != len(self.target_vecs):
            raise ValueError(f"Incompatible vocab sizes {len(self.source_vecs)}, {len(self.target_vecs)}")
        return len(self.source_vecs)
