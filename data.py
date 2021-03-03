import datetime
import json
import os

import numpy as np
import torch

from rossler_map import RosslerMap


class FastTensorDataLoader:
    def __init__(
        self, datatensor, dataset_len=100, batch_size=32, timesteps=2, shuffle=True, pre=True,
    ):
        self.tensors = datatensor  # (n_series, Niter, dim)
        self.dataset_len = dataset_len  # number of points of this dataset
        self.batch_size = batch_size
        self.timesteps = timesteps  # time dimension of output
        self.shuffle = shuffle
        self.pre = pre  # preshuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

        if (
            self.shuffle and self.pre
        ):  # select a random start position and a random series for each element on each batch
            self.idxs = np.random.randint(
                0,
                self.tensors.shape[1] - self.timesteps,
                size=(self.dataset_len, self.batch_size),
            )
            self.series_idxs = np.random.randint(
                0, self.tensors.shape[0], size=(self.dataset_len, self.batch_size)
            )

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.shuffle and self.pre:
            idx = self.idxs[self.i]
            series_idx = self.series_idxs[self.i]
        elif self.shuffle:
            idx =  np.random.randint(
                0,
                self.tensors.shape[1] - self.timesteps,
                size=(self.batch_size),
            )
            series_idx = np.random.randint(
                0, self.tensors.shape[0], size=(self.batch_size)
            )
        else:
            idx = self.i // self.dataset_len
            series_idx = np.arange(self.batch_size) // self.tensors.shape[0]
        batch = torch.stack(
            tuple(
                self.tensors[series_idx][idx : idx + self.timesteps]
                for idx, series_idx in zip(idx, series_idx)
            )
        ).double()
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def create_data(n_series=1, Niter=10000, delta_t=1e-2):
    """ Creates data matrix of size (n_series, Niter, dimensions) from n_series
    different sequences.

    Args:
        n_series (int, optional): [description]. Defaults to 1.
        Niter (int, optional): [description]. Defaults to 10000.
        delta_t ([type], optional): [description]. Defaults to 1e-2.
    """
    data = np.empty((n_series, Niter, 3))  # dimension = 3 because xyz
    noises = []
    for i in range(n_series):
        ROSSLER_MAP = RosslerMap(delta_t=delta_t)
        noise = np.random.randn(1)[0] * 1e-9
        noises.append(noise)
        disturbed_init = np.array([-5.75, -1.6, 0.02 + noise])
        traj, t = ROSSLER_MAP.full_traj(Niter, disturbed_init)
        data[i] = traj

    nowstr = str(datetime.datetime.now()).replace(" ", "_")
    os.makedirs(f"train_data/{nowstr}", exist_ok=True)
    np.save(f"train_data/{nowstr}/series.npy", data)
    with open(f"train_data/{nowstr}/meta.txt", "w") as file:
        file.write(
            json.dumps(
                {
                    "n_series": n_series,
                    "delta_t": delta_t,
                    "Niter": Niter,
                    "noise": noises,
                    "t_start": t[0],
                    "t_end": t[-1],
                }
            )
        )


def load_data(train_dir="train_data", dirname=None):
    dirnames = os.listdir(train_dir)
    if dirname is None:
        dirname = sorted(dirnames)[-1]
    data = np.load(os.path.join(train_dir, dirname, "series.npy"))
    data = torch.from_numpy(data)
    with open(os.path.join(train_dir, dirname, "meta.txt"), "r") as f:
        meta = json.load(f)
    return data, meta


SEED = 42
n_series = 11
Niter = 10000
delta_t = 0.01
if __name__ == "__main__":
    np.random.seed(SEED)
    create_data(n_series=n_series, Niter=Niter, delta_t=delta_t)

    # # test this if you want
    # data, meta = load_data()
    # dataloader = FastTensorDataLoader(
    #     data, dataset_len=10, batch_size=1, timesteps=15, shuffle=True
    # )

    # # visualize some data (beware of the sizes, keep them low)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # colors = [(1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5)]
    # for i, batch in enumerate(dataloader):
    #     for element in batch:
    #         # breakpoint()
    #         for j in range(3):
    #             plt.plot(
    #                 np.arange(1, len(element)), element[:-1, j], "-.", color=colors[j]
    #             )
    #             plt.plot(len(element), element[-1][None, j], "x", color=colors[j])
    # plt.show()

