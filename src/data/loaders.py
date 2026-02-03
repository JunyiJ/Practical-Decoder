import tiktoken
import torch
from torch.utils.data import Dataset


class TinyDataLoader:
    def __init__(self, data_path, batch_size, block_size, device):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch:i for i, ch in enumerate(chars)}
        self.itos = {i:ch for i, ch in enumerate(chars)}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

    def get_batch(self, split='train'):
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.block_size - 1
        if max_start <= 0:
            raise ValueError("Dataset is too small for the configured block_size")
        ix = torch.randint(max_start, (self.batch_size,))

        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x.to(self.device), y.to(self.device)

    def decode(self, ids):
        return "".join([self.itos[int(i)] for i in ids])

class TextDataset(Dataset):
    def __init__(self, data_path, batch_size, block_size, device):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.enc = tiktoken.get_encoding("gpt2")
        self.data = torch.tensor(self.enc.encode(text), dtype=torch.long)
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x.to(self.device), y.to(self.device)

    @property
    def vocab_size(self):
        return self.enc.n_vocab

    def get_batch(self, split="train"):
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.block_size - 1
        if max_start <= 0:
            raise ValueError("Dataset is too small for the configured block_size")
        ix = torch.randint(max_start, (self.batch_size,))

        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x.to(self.device), y.to(self.device)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.enc.decode(ids)


