import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yaml

from tqdm import tqdm

def process_data(preprocess_dir, files, labels, fq, channels, numTime, numBand, bands, key_prefix, sessions, persons, sectors):
    de = np.zeros([0, channels, numTime, numBand])
    emo_labels = np.empty([0, numTime], dtype=np.int32)
    subject_labels = np.empty([0, numTime], dtype=np.int32)

    total_files = files.shape[0]

    for f in tqdm(range(total_files), desc="Processing files"):
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

        file_de = 0.5 * np.log(file_de) + 0.5 * np.log(2 * np.pi * np.e * np.reciprocal(file_n))
        de = np.append(de, file_de[np.newaxis, :, :, :], axis=0)
        emo_labels = np.append(emo_labels, np.asarray([epoch_time_labels[int(t_[i] * fq)] for i in range(t_.shape[0])])[np.newaxis, :], axis=0)
        subject_labels = np.append(subject_labels, np.full([1, numTime], f // sessions, dtype=np.int32), axis=0)

    emo_labels = emo_labels + 1
    subject_labels = subject_labels + 1
    subject_labels = subject_labels - 8

    return de, emo_labels, subject_labels


def save_data(filename, de, emo_labels, subject_labels):
    with open(filename, 'wb') as f:
        np.save(f, de)
        np.save(f, emo_labels)
        np.save(f, subject_labels)

    return de, emo_labels, subject_labels

def split_data(de, emo_labels, subject_labels, test_ratio=0.1):
    de_train, de_test, emo_label_train, emo_label_test, subject_label_train, subject_label_test = \
        train_test_split(de, emo_labels, subject_labels, test_size=test_ratio, random_state=42)
    return de_train, de_test, emo_label_train, emo_label_test, subject_label_train, subject_label_test

def load_split_data(split_path='data_split.npz'):

    data_split = np.load(split_path)
    de_train = data_split['de_train']
    de_test = data_split['de_test']
    emo_label_train = data_split['emo_label_train']
    emo_label_test = data_split['emo_label_test']
    subject_label_train = data_split['subject_label_train']
    subject_label_test = data_split['subject_label_test']
    
    return de_train, de_test, emo_label_train, emo_label_test, subject_label_train, subject_label_test

def main(config):
    preprocess_dir = config['preprocess_dir']
    fq = config['fq']
    channels = config['channels']
    persons = config['persons']
    sessions = config['sessions']
    sectors = config['sectors']
    numTime = config['numTime']
    numBand = config['numBand']
    bands = config['bands']
    key_prefix = config['key_prefix']
    save_path = config['save_path']

    files = os.listdir(preprocess_dir)
    files.sort()
    files = np.asarray(files)[: sessions * persons]

    labels = scipy.io.loadmat(preprocess_dir + '/label.mat')['label'][0]

    de, emo_labels, subject_labels = process_data(preprocess_dir, 
                                                  files, labels, 
                                                  fq, channels, 
                                                  numTime, numBand, 
                                                  bands, key_prefix, 
                                                  sessions, persons, sectors)

    save_data(save_path+'/de.npy', de, emo_labels, subject_labels)

    de_train, de_test, emo_label_train, emo_label_test, subject_label_train, subject_label_test = \
        split_data(de, emo_labels, subject_labels, test_ratio=0.1)

    # 保存划分后的数据集
    np.savez(save_path+'/split_data.npz', 
             de_train=de_train, 
             de_test=de_test, 
             emo_label_train=emo_label_train, 
             emo_label_test=emo_label_test, 
             subject_label_train=subject_label_train, 
             subject_label_test=subject_label_test)

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)
