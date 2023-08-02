import os
import mne
import scipy
import pandas as pd
import numpy as np
import cebra
from cebra import CEBRA
import cebra.models
import matplotlib.pyplot as plt

preprocess_dir = '../../data/SEED/SEED_EEG/Preprocessed_EEG'
fq = 200
channels = 62
persons = 15
sessions = 3
sectors = 15

numTime = 5302
numBand = 5
bands = [1, 4, 8, 14, 31, 50]

key_prefix = ['ww', 'ww', 'ww', 'wsf', 'wsf', 'wsf', 'wyw', 'wyw', 'wyw',
              'xyl', 'xyl', 'xyl', 'ys', 'ys', 'ys', 'zjy', 'zjy', 'zjy',
              'djc', 'djc', 'djc', 'jl', 'jl', 'jl', 'jj', 'jj', 'jj',
              'lqj', 'lqj', 'lqj', 'ly', 'ly', 'ly', 'mhw', 'mhw', 'mhw',
              'phl', 'phl', 'phl', 'sxy', 'sxy', 'sxy', 'wk', 'wk', 'wk']

files = os.listdir(preprocess_dir)
files.sort()
files = np.asarray(files)[: sessions * persons]

labels = scipy.io.loadmat(preprocess_dir + '/label.mat')['label'][0]
de = np.zeros([0, channels, numTime, numBand])
emo_labels = np.empty([0, numTime], dtype=np.int32)
subject_labels = np.empty([0, numTime], dtype=np.int32)

# per file
for f in range(files.shape[0]):  
    eegs = scipy.io.loadmat(f'{preprocess_dir}/{files[f]}')
    epochs_time = np.empty([0, channels])
    epoch_time_labels = np.empty([0, ], dtype=np.int32)
    # per sector
    for i in range(1, sectors + 1):
        k = f'{key_prefix[f]}_eeg{str(i)}'
        epoch = eegs[k].swapaxes(0, 1)
        label = np.full((epoch.shape[0], ), labels[i - 1], dtype=np.int32)
        epochs_time = np.append(epochs_time, epoch, axis=0)
        epoch_time_labels = np.append(epoch_time_labels, label)
    epochs_time = epochs_time.swapaxes(0, 1)

    file_de = np.zeros([channels, numTime, numBand])
    file_n = np.zeros([channels, numTime, numBand])
    # per channel
    for i in range(channels):        
        session_spec, f_, t_, im = plt.specgram(epochs_time[i, :], Fs=fq)
        session_spec = session_spec.swapaxes(0, 1)       
        bins = np.digitize(f_, bands)
        
        for t in range(session_spec.shape[0]):
            for j in range(session_spec.shape[1]):
                if bins[j] > 0 and bins[j] <= numBand:
                    file_de[i][t][bins[j] - 1] += session_spec[t][j] ** 2
                    file_n[i][t][bins[j] - 1] += 1
    
    # for i in range(channels):
    #     for t in range(numTime):
    #         for j in range(numBand):
    #                 print(file_n[i][t][j])
    
    file_de = 0.5 * np.log(file_de) + 0.5 * np.log(2 * np.pi * np.e * np.reciprocal(file_n))
    de = np.append(de, file_de[np.newaxis, :, :, :], axis=0)
    emo_labels = np.append(emo_labels, 
                            np.asarray([epoch_time_labels[(int)(t_[i] * fq)] 
                                        for i in range(t_.shape[0])])[np.newaxis, :], axis=0)
    subject_labels = np.append(subject_labels,
                               np.full([1, numTime], f // sessions, dtype=np.int32), axis=0)

emo_labels = emo_labels + 1
subject_labels = subject_labels + 1
subject_label = subject_label - 8
#print(emo_labels.shape, subject_labels.shape)

with open('de.npy', 'wb') as f:
    np.save(f, de)
    np.save(f, emo_labels)
    np.save(f, subject_labels)


with open('de.npy', 'rb') as f:
    de = np.load(f)
    emo_labels = np.load(f)
    subject_labels = np.load(f)

from sklearn.model_selection import train_test_split
test_ratio = 0.1
de_train, de_test, emo_label_train, emo_label_test, subject_label_train, subject_label_test = \
    train_test_split(de, emo_labels, subject_labels, 
                   test_size=test_ratio, 
                   random_state=42)
#print(de_train.shape, de_test.shape, emo_label_train.shape, emo_label_test.shape)