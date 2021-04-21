import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import read_scp, read_wav
import yaml
import matplotlib.pyplot as plt


class wav_dataset(Dataset):
    def __init__(self,config,path_scp_mix,path_scp_targets):

        self.scp_mix = read_scp(path_scp_mix)
        self.scp_targets = [read_scp(path_scp_target_i) \
                                for path_scp_target_i in path_scp_targets]

        self.keys = [key for key in self.scp_mix.keys()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]

        y_mix = read_wav(self.scp_mix[key])
        y_targets = [read_wav(scp_target_i[key]) \
                                for scp_target_i in self.scp_targets]
        
        return y_mix,y_targets


def padding(batch):
    batch_mix, batch_s = [],[]
    for mix, s in batch:
        batch_mix.append(torch.tensor(mix,dtype=torch.float32).unsqueeze(-1))
        batch_s.append(torch.tensor(s,dtype=torch.float32).transpose(0,1))

    batch_mix = pad_sequence(batch_mix, batch_first=True)
    batch_size = batch_mix.shape[0]
    batch_mix = batch_mix.view([batch_size,-1])

    batch_s = pad_sequence(batch_s, batch_first=True)

    return batch_mix, batch_s


def make_dataloader(config, path_scp_mix, path_scp_targets):
    batch_size = config['dataloader']['batch_size']
    num_workers = config['dataloader']['num_workers']
    shuffle = config['dataloader']['shuffule']
    dataset  = wav_dataset(config, path_scp_mix, path_scp_targets)

    dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,
                                shuffle=shuffle,collate_fn=padding)

    return dataloader


if __name__ == "__main__":
    with open('./config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
    m = make_dataloader(config, "./scp/tt_mix.scp", ["./scp/tt_s1.scp","./scp/tt_s2.scp"])
    for yaa,hin in m:
        print(yaa.shape)
