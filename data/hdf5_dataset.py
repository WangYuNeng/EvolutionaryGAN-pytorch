from data.base_dataset import BaseDataset, get_transform
import h5py as h5
import os
from PIL import Image


class Hdf5Dataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--hdf5_filename', type=str, default='img_align_celeba_128.hdf5',
                            help='the name of hdf5 file')
        parser.add_argument('--load_in_mem', action='store_true', default=False,
                            help='Load all data into memory? (default: %(default)s)')
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
        # get the image paths of your dataset;
        self.hdf5_path = os.path.join(opt.dataroot, opt.hdf5_filename)
        self.load_in_mem = opt.load_in_mem
        self.imkey = None
        self.lkey = None

        with h5.File(self.hdf5_path, 'r') as f:
            key_list = list(f.keys())
            for key in key_list:
                if key == 'data' or key == 'imgs':
                    self.imkey = key
                    self.num_imgs = len(f[self.imkey])
                elif key == 'label' or key == 'labels':
                    self.lkey = key
                else:
                    raise ValueError('Unkown key in the HDF5 file.')

            # If loading into memory, do so now
            if self.load_in_mem:
                print('Loading %s into memory...' % self.hdf5_path)
                self.data = f[self.imkey][:]
                self.labels = f[self.lkey][:] if (self.lkey is not None) else None

        # define the default transform function.
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        """
        if self.load_in_mem:
            img = self.data[index]
            label = self.labels[index] if (self.lkey is not None) else -1
        else:
            with h5.File(self.hdf5_path, 'r') as f:
                img = f[self.imkey][index]
                label = f[self.lkey][index] if (self.lkey is not None) else -1
        if img.shape[0] <= 3:
            img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)
        img = self.transform(img)

        return {'data': img, 'condition': label}

    def __len__(self):
        """Return the total number of images."""
        return self.num_imgs
