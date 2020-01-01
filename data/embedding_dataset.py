import os
import subprocess
import random
from joblib import Parallel, delayed

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from data.base_dataset import BaseDataset


load_dotenv('./.env')
DATA_ROOT = os.environ.get('EMBEDDING_PATH')


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
        '0to0.1.word2vec-cbow.vec': 'https://drive.google.com/open?id=1Y7aNqEbyB9nrbLExOb6BTn8po3234Qrb',
        '0.1to0.2.word2vec-cbow.vec': 'https://drive.google.com/open?id=1PRnDSx42YFIw5Jnj5mnTWeTJmmQvueKA',
        '0to0.1.word2vec-skipgram.vec': 'https://drive.google.com/open?id=11iZVCbY-qwQZBdiEyjpt-5XdsPZsDHEP',
        '0.1to0.2.word2vec-skipgram.vec': 'https://drive.google.com/open?id=1PeB4Lyx_xLQCV9wHJWs2PGsJ7zpRqZ6k',
        '0to0.1.glove.vec': 'https://drive.google.com/open?id=1atZnVBUqaN9zOfTpilTbV0QCmS8mbYKF',
        '0.1to0.2.glove.vec': 'https://drive.google.com/open?id=1AmRZy-tE8YJbU-qYEhK7OPzyYvFo6xj4',

        '0to0.1.cbow10.vec': 'https://drive.google.com/open?id=1TrfIf3VzHXv0RKKhdsh6LOdJiDUIK1nd',
        '0.1to0.2.cbow10.vec': 'https://drive.google.com/open?id=1Pd_bfTHJj2KvG7APvD_UMrwD9KSSdPlz',
        '0to0.1.word2vec-cbow10.vec': 'https://drive.google.com/open?id=17KXvWSuUhJ46d81ixkVlLDwOBkWNnaaH',
        '0.1to0.2.word2vec-cbow10.vec': 'https://drive.google.com/open?id=18Ulj6d6TVLTP0RDNiT4u179aVcbGJoAL',
        '0to0.1.skipgram10.vec': 'https://drive.google.com/open?id=1O1UMRTp8NQj-IUi05OizBSiGXj6BTfTC',
        '0.1to0.2.skipgram10.vec': 'https://drive.google.com/open?id=1c6ruZxXt7BawY0RStXYa2SHRbmUvAB3O',
        '0to0.1.word2vec-skipgram10.vec': 'https://drive.google.com/open?id=1AFqIJVpBGHcCDvEDxPW6RRuI4UY9zdzT',
        '0.1to0.2.word2vec-skipgram10.vec': 'https://drive.google.com/open?id=1cPwEwXISFmO1Uif6h4Q8kFyl8pvblmE9',
        '0to0.1.glove10.vec': 'https://drive.google.com/open?id=1rPaEAc756yWjd68cq2bRJA4OfJ4uYh5l',
        '0.1to0.2.glove10.vec': 'https://drive.google.com/open?id=1WmArLQ_3qbOt6N4LT8b0GwZ-IzOWC8ha',
    }

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--source_dataset_name', type=str, default=None,
                            help='name of imported dataset, options: [cbow, fasttext]')
        parser.add_argument('--target_dataset_name', type=str, default=None,
                            help='name of imported dataset, options: [cbow, fasttext]')
        parser.add_argument('--exact_orthogonal', action='store_true')
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
        self.data_root = DATA_ROOT
        if not os.path.isdir(self.data_root):
            os.mkdir(self.data_root)

        self.source_name = opt.source_dataset_name
        self.target_name = opt.target_dataset_name
        self.normalize_mode = opt.preprocess
        self.source_url_name, self.target_url_name = self.get_url_names(
            self.source_name,
            self.target_name,
            list(self.download_urls.keys()),
        )
        assert self.source_name != 'mock'

        for url_name in [self.source_url_name, self.target_url_name]:
            self.download_embeddings(url_name)

        self.source_vecs, self.source_word2idx, self.source_idx2word = self.load_embeddings(
            self.source_url_name,
            self.opt.max_dataset_size,
        )
        if self.target_url_name == 'mock':
            self.target_idx2word, self.target_word2idx = self.source_idx2word, self.source_word2idx
            self.target_vecs, _ = self.create_mock_target(self.source_vecs)
        else:
            self.target_vecs, self.target_word2idx, self.target_idx2word = self.load_embeddings(
                self.target_url_name,
                self.opt.max_dataset_size,
            )

    @staticmethod
    def create_mock_target(source_vecs):
        emb_dim = source_vecs.shape[-1]
        random_mat = np.random.random([emb_dim, emb_dim]).astype(source_vecs.dtype)
        triu = np.triu(random_mat)
        skew = triu - triu.T
        I = np.identity(emb_dim, dtype=source_vecs.dtype)
        mapping = (I - skew) @ np.linalg.inv(I + skew)
        target_vecs = source_vecs @ mapping
        return target_vecs, mapping

    @staticmethod
    def get_url_names(source_name: str, target_name: str, url_names: list):
        matched_url_names = []
        for match_name in [source_name, target_name]:
            if match_name == 'mock':
                matched_url_names.append('mock')
                continue
            for name in url_names:
                if match_name in name:
                    matched_url_names.append(name)
                    url_names.remove(name)
                    break
        return matched_url_names

    def download_embeddings(self, url_name: str):
        if url_name == 'mock':
            return
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
        vecs = self.normalize_vecs(vecs, self.normalize_mode)
        word2idx = {w: i for i, w in enumerate(words)}
        idx2word = {i: w for i, w in enumerate(words)}
        if not len(words) == min(vocab_size, max_vocab_size) or not vecs.shape[1] == emb_dim:
            raise ValueError(
                f'corrupted embedding {file_path},'
                f'vecs.shape = {vecs.shape}'
            )
        return vecs, word2idx, idx2word

    @staticmethod
    def normalize_vecs(vecs: np.array, normalize_mode: str):
        """
        Normalize embeddings by their norms / recenter them.
        """
        for t in normalize_mode.split(','):
            if t == '':
                continue
            if t == 'center':
                mean = vecs.mean(0, keepdims=True)
                vecs -= mean
            elif t == 'renorm':
                vecs /= vecs.norm(2, 1, keepdims=True)
            else:
                raise Exception('Unknown normalization type: "%s"' % t)
        return vecs

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
        return min(len(self.target_vecs), self.opt.most_frequent)
