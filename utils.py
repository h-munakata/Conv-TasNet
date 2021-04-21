import os
import sys
import soundfile as sf
import numpy as np
import librosa


def read_scp(scp_path):
    files = open(scp_path, 'r')
    lines = files.readlines()
    scp_wav = {}
    for line in lines:
        line = line.split()
        if line[0] in scp_wav.keys():
            raise ValueError
        scp_wav[line[0]] = line[1]
    return scp_wav


def read_wav(path_wav):
    y,_ = sf.read(path_wav)
    return y

def write_wav(path_wav, y, sr):
    os.makedirs(os.path.dirname(path_wav), exist_ok=True)
    y = np.array(y)
    y = 0.9/np.max(y) * y
    sf.write(path_wav, y, sr)

def stft(s,n_fft=512):
    S = librosa.stft(s, n_fft=n_fft, hop_length=n_fft//4,
                            win_length=n_fft//2, window='hann',
                            center=False)
    return S

def istft(S,n_fft=512):
    s = librosa.istft(S, hop_length=n_fft//4, win_length=n_fft//2,
                            window='hann',center=False)
    return s


def save_scp(dir_wav,scp_name):
    os.makedirs("./scp", exist_ok=True)
    path_scp = "./scp" + "/" + scp_name

    print("making {0} from {1}".format(path_scp,dir_wav))

    if not os.path.exists(dir_wav):
        raise ValueError("directory of .wav doesn't exist")

    with open(path_scp,'w') as scp:
        for root, dirs, files in os.walk(dir_wav):
            files.sort()
            for file in files:
                scp.write(file+" "+root+'/'+file)
                scp.write('\n')


def wav2scp(dir_dataset,num_spks):
    print('making scp files')

    type_list = ['/tr','/cv', '/tt']

    for type_data in type_list:
        dir_type = dir_dataset + type_data + '/mix'
        scp_name = type_data + '_mix.scp'
        save_scp(dir_type,scp_name)

        for i in range(num_spks):
            dir_type = dir_dataset + type_data + '/s{0}'.format(i+1)
            scp_name = type_data + '_s{0}.scp'.format(i+1)
            save_scp(dir_type,scp_name)
            

if __name__ == "__main__":
    dir_wav = "/data1/h_munakata/wsj0/2speakers/min"
    wav2scp(dir_wav,2)