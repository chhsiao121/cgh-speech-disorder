import os
import numpy as np
from multiprocessing import Pool
import joblib
from numpy.lib.function_base import append
import scipy as sp

num_threads = 28
feature_size=128
PATH = '/home/angel00540/work_/work/cgh_2022/exp_0603/data_syllable/training/adult/'

IN_PATH = []


def save_spectrum_to_npy(wavfile):
    import librosa
    y, _ = librosa.load(wavfile)
    S = np.abs(librosa.stft(y, n_fft=512))
    p = librosa.amplitude_to_db(S, ref=np.max)
    tmp = np.zeros([256, 128])
    if p.shape[1] > 128:
        # print(p.shape[1])
        tmp[:256, :128] = p[:256, :128]
    else:
        tmp[:256, :p.shape[1]] = p[:256, :p.shape[1]]
    tmp = (tmp+40)
    tmp = tmp/40.0
    np.save(wavfile[:-4]+'.npy', tmp)


def save_mfcc_to_npy(wavfile):
    import librosa
    y, sr = librosa.load(wavfile)
    data_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
    # print(data_mfcc.shape)
    # len_dd.append(data_mfcc.shape[1])

    tmp = np.zeros([32, 64])
    if data_mfcc.shape[1] > 64:
        # print(data_mfcc.shape[1])
        tmp[:32, :64] = data_mfcc[:32, :64]
    else:
        tmp[:32, :data_mfcc.shape[1]] = data_mfcc[:64, :data_mfcc.shape[1]]
    # tmp = (tmp+40)
    # tmp = tmp/40.0
    np.save(wavfile[:-4]+'.npy', tmp)


def save_3ch_spectrum(wavfile):
    import librosa
    from skimage.transform import resize

    fix_n_sec = 2
    mini_sec =0.3
    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]
    y=[]
    try:
        y, sr = librosa.load(wavfile)
    except:
        print("read error: ",wavfile)

    if len(y)>0:
        # pad if len less 1 sec
        if len(y)/sr > fix_n_sec:
            x = y[:fix_n_sec * sr]
            run=1
        elif mini_sec < len(y)/sr < fix_n_sec :
            z = int(fix_n_sec*sr - len(y))
            if(z%2==0):
                z1 = int(z/2)
                z2 = int(z/2)
            else:
                z1 = int((z-1)/2)
                z2 = int(z1 + 1)
            try:
                x = np.pad(y,(z1,z2), 'linear_ramp', end_values=(0, 0))
                run=1
            except:
                print("pad error: ",wavfile)
            # ([0]*append_zero)+y+([0]*append_zero)
        elif len(y)/sr < mini_sec:
            print('len < '+mini_sec+': ',wavfile)
            run=0

        if run==1:
            # audio normalizedy
            normalizedy = librosa.util.normalize(x)

            specs = []
            for i in range(num_channels):

                window_length = int(round(window_sizes[i]*sr/1000))
                # print('win_len: ',window_length)
                hop_length = int(round(hop_sizes[i]*sr/1000))
                mel = librosa.feature.melspectrogram(
                    y=normalizedy, sr=sr,n_fft=window_length,hop_length=hop_length, win_length=window_length)
                mellog = np.log(mel + 1e-9)

                spec = librosa.util.normalize(mellog)

                spec = resize(mellog, (128, feature_size))
                spec = np.asarray(spec)
                specs.append(spec)

            # list to np array
            specs = np.asarray(specs)
            specs = np.moveaxis(specs,0,2)
            print('specs.shape : ',specs.shape)
            np.save(wavfile[:-4]+'.npy', specs)


def print_len(wavfile):
    import librosa
    try:
        y, sr = librosa.load(wavfile)
        return len(y)/sr

    except:
        return 'error: ',wavfile

for root2, dirs2, files2 in os.walk(os.path.abspath(PATH)):
    for file2 in files2:
        if('wav' in file2):
            IN_PATH.append(os.path.join(root2, file2))


print(len(IN_PATH))
n_jobs=num_threads
verbose=1
jobs = [ joblib.delayed(save_3ch_spectrum)(i) for i in IN_PATH ]
joblib.Parallel(n_jobs=28,verbose=verbose)(jobs)
    