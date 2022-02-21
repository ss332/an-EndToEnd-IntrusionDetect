import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class IDSDataset(Dataset):

    def __init__(self, annotations_file, img_dir=r'', transform=None, target_transform=None):
        self.sessions_labels = pd.read_csv(annotations_file, usecols=['id', 'file', 'label'])
        print(self.sessions_labels['label'].value_counts())
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sessions_labels)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.sessions_labels.iloc[idx, 1],
                                r'session{}.pt'.format(self.sessions_labels.iloc[idx, 0]))
        session = torch.load(img_path)
        label = self.sessions_labels.iloc[idx, 2]
        label = torch.tensor(label)
        if self.transform:
            image = self.transform(session)
        if self.target_transform:
            label = self.target_transform(label)
        packets, p_len = self.encode_packets(session)
        en_session = self.encode_sessions(session)

        return en_session, packets, p_len, label

    def encode_packets(self, session):

        vec = torch.zeros(32, 320)
        l = session.size(0)
        if l >= 32:
            l = 32
        for i in range(l):
            vec[i] = session[i]
        return vec, torch.tensor(l)

    def encode_sessions(self, session):
        session_tensor = torch.zeros(1024)
        size = 0
        for k in range(session.size(0)):
            x = session[k]
            for j in range(x.size(0)):
                if x[j] != 0 and size < 1024:
                    session_tensor[size] = x[j]
                    size = size + 1

            # (n,c,w,h)
        session_tensor = session_tensor.view(1, 32, 32)
        return session_tensor
