import os
import pickle
import lmdb
import torch
from torch.utils.data import Dataset

class LigandDataset(Dataset):

    def __init__(self, lmdb_path, transform=None):
        super().__init__()
        self.processed_path = lmdb_path
        self.transform = transform
        self.db = None
        self.keys = None
        
    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        with self.db.begin() as txn: 
            key = self.keys[idx]
            data = pickle.loads(txn.get(key))
            data.id = idx
            if self.transform is not None:
                data = self.transform(data)
            if data.ligand_context_pos.shape[0] == 0:
                return self.__getitem__((idx + 1) % len(self.keys))
        # key = self.keys[idx]
        # data = pickle.loads(self.db.begin().get(key))
        # data.id = idx
        # if self.transform is not None:
        #     data = self.transform(data)

        # if data.ligand_context_pos.shape[0] == 0:
        #     return self.__getitem__((idx + 1))
        return data
    
    def __del__(self):
        self._close_db()