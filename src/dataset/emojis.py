import h5py
import numpy as np
from datetime import datetime


class EmojiDataset:
    def __init__(self, dataset_path, batch_size: int = 8, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            seed = int(datetime.now().timestamp())
            rng = np.random.default_rng(seed=seed)

        with h5py.File(dataset_path, "r") as f:
            assert 'images' in f and 'token_embeddings' in f
            dataset_length = len(f['images'])  # type: ignore
            prompt_length, token_dim = f['token_embeddings'].shape[-2:]  # type: ignore

        self.dataset_path = dataset_path
        self.rng = rng
        self.batch_size = batch_size
        self.dataset_length = dataset_length
        self.prompt_length = prompt_length
        self.token_dim = token_dim
        self.file = None

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.dataset_path, 'r')

        idx = sorted(idx)  # h5 access must be done using sorted indices.
        images = np.array(self.file['images'][idx])  # type: ignore
        prompts = np.array(self.file['token_embeddings'][idx])  # type: ignore

        return prompts, images

    def sample_batch(self):
        idx = self.rng.choice(self.dataset_length, size=self.batch_size, replace=False)
        return self[idx]
