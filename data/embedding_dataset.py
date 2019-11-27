import os
import subprocess
import random
from joblib import Parallel, delayed

import numpy as np
from tqdm import tqdm

from data.base_dataset import BaseDataset


class EmbeddingDataset(BaseDataset):
    """
    pretrained word embedding are at
    https://drive.google.com/drive/folders/1CuI62zaN1TUc-JZA_MEVAQngP7fJR89c
    """
    download_urls = {
        '0to0.1.cbow.vec': 'https://drive.google.com/open?id=1rjsPNhAovS5Sx9AocUiEJsjzyPXdov4g',
        '0.1to0.2.cbow.vec': 'https://drive.google.com/open?id=1jVp8Jtqg5l03TokHEY1Mn91zb549Xy8V',
        '0to0.1.skipgram.vec': 'https://drive.google.com/open?id=1xNJC-l_iQuL9CVotfaA25sKXESpV1U6O',
        '0.1to0.2.skipgram.vec': 'https://drive.google.com/open?id=1RGMshfU03ZTLFPf_qqxlYAsGP18m2Nhw',
        '0to0.1.glove.vec': 'https://drive.google.com/open?id=1atZnVBUqaN9zOfTpilTbV0QCmS8mbYKF',
        '0.1to0.2.glove.vec': 'https://drive.google.com/open?id=1Hl-OiRB6M8XfsQW2sd9UOZPTk_1DbSNc',
    }

    @staticmethod
    def modify_commandline_options(parser, is_train):
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

        self.source_vecs, self.source_word2idx, self.source_idx2word = self.load_embeddings(
            self.source_url_name,
            self.opt.max_vocab_size,
        )
        self.target_vecs, self.target_word2idx, self.target_idx2word = self.load_embeddings(
            self.target_url_name,
            self.opt.max_vocab_size,
        )

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

    def load_embeddings(self, url_name: str, max_vocab_size=None):
        print(f'Loading {url_name}...')
        file_path = os.path.join(self.data_root, url_name)
        with open(file_path, 'r') as f:
            vocab_size, emb_dim = [int(i) for i in f.readline().split()]
            embedding_index = dict(
                Parallel(n_jobs=-1)(
                    delayed(self.load_line_from_file)(next(f))
                    for _ in tqdm(range(min(vocab_size, max_vocab_size)))
                )
            )
        words = embedding_index.keys()
        vecs = np.asarray(list(embedding_index.values()))
        vecs = (vecs - np.mean(vecs, axis=1, keepdims=True)) / np.std(vecs, axis=1, keepdims=True)  # normalize
        word2idx = {w: i for i, w in enumerate(words)}
        idx2word = {i: w for i, w in enumerate(words)}
        if not len(words) == min(vocab_size, max_vocab_size) or not vecs.shape[1] == emb_dim:
            raise ValueError(
                f'corrupted embedding {file_path},'
                f'vecs.shape = {vecs.shape}'
            )
        return vecs, word2idx, idx2word

    @staticmethod
    def load_line_from_file(line):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        return word, coefs

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        """
        target_index = (index + random.randint(0, self.__len__())) % self.__len__()
        return {'data': self.target_vecs[target_index], 'source': self.source_vecs[index], 'source_idx': index}

    def __len__(self):
        """Return the total number of word-vectors."""
        return len(self.target_vecs)
