from network import ConvTasNet
import os
import yaml
from dataloader import wav_dataset
import torch
from eval_utils import eval_each_s
from utils import write_wav
from tqdm import tqdm
import datetime
import csv
import sys

class Separation():
    def __init__(self, path_model):
        path_config = os.path.join(os.path.basename(path_model), 'config.yaml')
        with open('./config.yaml', 'r') as yml:
            config = yaml.safe_load(yml)
        self.model = ConvTasNet(config)
        ckp = torch.load(path_model,map_location=torch.device('cpu'))
        self.model.load_state_dict(ckp['model_state_dict'])
        dt_now = datetime.datetime.now()
        time = str(dt_now.strftime('%Y-%m-%d-%H:%M:%S'))
        self.dir_save = './separated/'+time
        os.makedirs(self.dir_save,exist_ok=True)
        self.eval_dataset  = wav_dataset(config, "./scp/tt_mix.scp", ["./scp/tt_s1.scp","./scp/tt_s2.scp"])

    def run(self,save_sound=False):
        list_eval = []
        with open(os.path.join(self.dir_save,'result.csv'), 'w') as f:
            writer = csv.writer(f)
            # est_value, base_value, ideal_value, improve_value, improve_ideal_value
            header = ['SI-SDR_1','SI-SDR_2','Base SI-SDR_1','Base SI-SDR_2','IBM SI-SDR_1','IBM SI-SDR_2',
                    'SI-SDRi_1','SI-SDRi_2','IBM SI-SDRi_1','IBM SI-SDRi_2']
            writer.writerow(header)
            for idx in range(len(self.eval_dataset)):
                key = self.eval_dataset.keys[idx]
                filename = self.eval_dataset.keys[idx]
                mix_s,s = self.eval_dataset[idx]
                C = len(s)
                mix_s = torch.tensor(mix_s,dtype=torch.float32).reshape([1,-1])
                s = torch.tensor(s,dtype=torch.float32).T.reshape([1,-1,C]).detach()
                est_s = self.model(mix_s).detach()

                

                est_value, base_value, ideal_value, improve_value, improve_ideal_value = eval_each_s(est_s, s, mix_s)

                result = est_value + base_value + ideal_value + improve_value + improve_ideal_value
                list_eval.append(result)
                print("idx:{}. key:{}, SI-SDRi:{}, SI-SDR:{}, Base SI-SDR{}".format(idx,key, improve_value, est_value, base_value))

                if save_sound:
                    for c in range(C):
                        est_s_i = est_s[:,:,c].reshape(-1)
                        path_wav = os.path.join(self.dir_save,filename.replace('.wav','_') + str(c+1) +'.wav')
                        write_wav(path_wav,est_s_i,16000)

                writer.writerow(result)

    def separate(self):
        pass

            


if __name__ == "__main__":
    with open('./config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)

    path_model = sys.argv[1]
    if len(sys.argv>1):
        save_sound = sys.argv[2]
    else:
        save_sound = False
    separation = Separation(path_model)
    separation.run(save_sound=save_sound)