# Copyright 2020 Mickael Chen, Edouard Delasalles, Jean-Yves Franceschi, Patrick Gallinari, Sylvain Lamprier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

import numpy as np


# def load_dataset(config, train):
#     """
#     Loads a dataset.

#     Parameters
#     ----------
#     config : helper.DotDict
#         Configuration to use.
#     train : bool
#         Whether to load the training or testing dataset.

#     Returns
#     -------
#     data.base.VideoDataset
#         Dataset corresponding to the input configuration.
#     """
#     name = config.dataset
#     if name == 'smmnist':
#         from data.mmnist import MovingMNIST
#         return MovingMNIST.make_dataset(config.data_dir, config.nx, config.seq_len, config.max_speed,
#                                         config.deterministic, config.ndigits, train)
#     if name == 'kth':
#         from data.kth import KTH
#         return KTH.make_dataset(config.data_dir, config.nx, config.seq_len, train)
#     if name == 'human':
#         from data.human import Human
#         return Human.make_dataset(config.data_dir, config.nx, config.seq_len, config.subsampling, train)
#     if name == 'bair':
#         from data.bair import BAIR
#         return BAIR.make_dataset(config.data_dir, config.seq_len, train)
#     raise ValueError(f'No dataset named \'{name}\'')


# def collate_fn(videos):
#     """
#     Collate function for the PyTorch data loader.

#     Merges all batch videos in a tensor with shape (length, batch, channels, width, height) and converts their pixel
#     values to [0, 1].

#     Parameters
#     ----------
#     videos : list
#         List of uint8 NumPy arrays representing videos with shape (length, batch, width, height, channels).

#     Returns
#     -------
#     torch.*.Tensor
#         Batch of videos with shape (length, batch, channels, width, height) and float values lying in [0, 1].
#     """
#     seq_len = len(videos[0])
#     batch_size = len(videos)
#     nc = 1 if videos[0].ndim == 3 else 3
#     w = videos[0].shape[1]
#     h = videos[0].shape[2]
#     tensor = torch.zeros((seq_len, batch_size, nc, h, w), dtype=torch.uint8)
#     for i, video in enumerate(videos):
#         if nc == 1:
#             tensor[:, i, 0] += torch.from_numpy(video)
#         if nc == 3:
#             tensor[:, i] += torch.from_numpy(np.moveaxis(video, 3, 1))
#     tensor = tensor.float()
#     tensor = tensor / 255
#     return tensor


class VideoDataset(object):
    """
    Abstract class of a video dataset.

    Requires an attribute 'data' which is a list containing the dataset data.

    data : list
        List containing the dataset data.
    """
    def get_fold(self, fold):
        """
        Selects a chunk of the dataset by splitting the dataset data depending on the chosen fold.

        'test' does not change the dataset but only checks whether it is in test mode.

        When training, 'train' randomly selects 95% of the attribute list 'data' (corresponding to examples fed to the
        model during training) with respect to a fixed random seed, and 'val' selects the remaining 5% (corresponding
        to the validation set).

        Parameters
        ----------
        fold : str
            'train', 'val', or 'test', depending on the dataset chunk to select.

        Returns
        -------
        data.base.VideoDataset
            Same dataset with data restricted to the given fold.
        """
        if fold in ('train', 'val'):
            # Select 95% of the training dataset for training, and 5% for evaluation purposes
            assert self.train
            # Random selection with fixed seed
            rng = np.random.RandomState(42)
            rand_ids = list(range(len(self.data)))
            rng.shuffle(rand_ids)
            n_train = int(0.95 * len(rand_ids))
            folds = {
                'train': set(rand_ids[:n_train]),
                'val': set(rand_ids[n_train:])
            }
            data = [x for i, x in enumerate(self.data) if i in folds[fold]]
        else:
            assert fold == 'test' and not self.train
            data = self.data
        # Filter
        return self._filter(data)

    def change_seq_len(self, seq_len):
        """
        Changes the length of sequences in the dataset.

        Parameters
        ----------
        seq_len : int
            New sequence length.
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError

    def _filter(self, data):
        """
        Returns the same dataset with new data.

        Parameters
        ----------
        data : list
            List containing the new dataset data.

        Returns
        -------
        data.base.VideoDataset
            Same dataset with original data replaced by the input data.
        """
        raise NotImplementedError

    def collate_fn(self, videos):
        """
        Collate function for the PyTorch data loader.

        Merges all batch videos in a tensor with shape (length, batch, channels, width, height) and converts their pixel
        values to [0, 1].

        Parameters
        ----------
        videos : list
            List of uint8 NumPy arrays representing videos with shape (length, batch, width, height, channels).

        Returns
        -------
        torch.*.Tensor
            Batch of videos with shape (batch,length,channels, width, height) and float values lying in [0, 1].
        """
        seq_len = len(videos[0])
        batch_size = len(videos)
        nc = 1 if videos[0].ndim == 3 else 3
        w = videos[0].shape[1]
        h = videos[0].shape[2]
        tensor = torch.zeros((batch_size,seq_len, nc, h, w), dtype=self.dtype)
        for i, video in enumerate(videos):
            if nc == 1:
               # tensor[:, i, 0] += torch.from_numpy(video)
                tensor[i,:,0]  += torch.from_numpy(video)
            if nc == 3:
                tensor[:, i] += torch.from_numpy(np.moveaxis(video, 3, 1))
        tensor = tensor.type(self.dtype) #tensor.float()
        tensor = tensor / 255
        return tensor
