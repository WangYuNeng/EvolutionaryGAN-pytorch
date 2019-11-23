from collections import OrderedDict

import torch
import numpy as np

from util.inception import get_inception_score
from util.util import one_hot
from TTUR import fid
from inception_pytorch import inception_utils
from .base_evaluator import BaseEvaluator


class ImageEvaluator(BaseEvaluator):

    def __init__(self, opt, model, dataset):
        super().__init__(opt, model, dataset)
        # scores init
        if self.opt.use_pytorch_scores and self.opt.score_name is not None:
            no_FID = not('FID' in self.opt.score_name)
            no_IS = not('IS' in self.opt.score_name)
            parallel = len(opt.gpu_ids) > 1
            self.get_inception_metrics = inception_utils.prepare_inception_metrics(opt.dataset_name, parallel, no_IS,
                                                                                   no_FID)
        elif 'FID' in self.opt.score_name:
            STAT_FILE = self.opt.fid_stat_file
            INCEPTION_PATH = "./inception_v3/"

            print("load train stats.. ")
            # load precalculated training set statistics
            f = np.load(STAT_FILE)
            self.mu_real, self.sigma_real = f['mu'][:], f['sigma'][:]
            f.close()
            print("ok")

            inception_path = fid.check_or_download_inception(INCEPTION_PATH)  # download inception network
            fid.create_inception_graph(inception_path)  # load the graph into the current TF graph

            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            # self.sess = tf.Session(config = config)
            # self.sess.run(tf.global_variables_initializer())

    def get_current_scores(self):
        scores_ret = OrderedDict()
        return scores_ret
        # TODO
        # print("%d samples generating done" % self.opt.evaluation_size)
        # if self.opt.use_pytorch_scores:
        #     self.IS_mean, self.IS_var, self.FID = self.get_inception_metrics(
        #         samples,
        #         self.opt.evaluation_size,
        #         num_splits=10,
        #     )
        #     if 'FID' in self.opt.score_name:
        #         print(self.FID)
        #         scores_ret['FID'] = float(self.FID)
        #     if 'IS' in self.opt.score_name:
        #         print(self.IS_mean, self.IS_var)
        #         scores_ret['IS_mean'] = float(self.IS_mean)
        #         scores_ret['IS_var'] = float(self.IS_var)
        #
        # else:
        #     # Cast, reshape and transpose (BCHW -> BHWC)
        #     samples = samples.cpu().numpy()
        #     samples = ((samples + 1.0) * 127.5).astype('uint8')
        #     samples = samples.reshape(self.opt.evaluation_size, 3, self.opt.crop_size, self.opt.crop_size)
        #     samples = samples.transpose(0, 2, 3, 1)
        #     for name in self.opt.score_name:
        #         if name == 'FID':
        #             mu_gen, sigma_gen = fid.calculate_activation_statistics(samples,
        #                                                                     self.sess,
        #                                                                     batch_size=self.opt.fid_batch_size,
        #                                                                     verbose=True)
        #             print("calculate FID:")
        #             try:
        #                 self.FID = fid.calculate_frechet_distance(mu_gen, sigma_gen, self.mu_real, self.sigma_real)
        #             except Exception as e:
        #                 print(e)
        #                 self.FID = 500
        #             print(self.FID)
        #             scores_ret[name] = float(self.FID)
        #         if name == 'IS':
        #             Imlist = []
        #             for i in range(len(samples)):
        #                 im = samples[i, :, :, :]
        #                 Imlist.append(im)
        #             print(np.array(Imlist).shape)
        #             self.IS_mean, self.IS_var = get_inception_score(Imlist)
        #
        #             scores_ret['IS_mean'] = float(self.IS_mean)
        #             scores_ret['IS_var'] = float(self.IS_var)
        #             print(self.IS_mean, self.IS_var)
        #
        # return scores_ret
