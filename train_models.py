import os
import yaml
import time
import logging
import cebra
from cebra import CEBRA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import extract_de


def train_cebra_models(models, de_train, emo_label_train, subject_label_train, 
                        numTime, batch_size, max_iter, embedding_dimensions):


    for model_name, offset in models:
        for d in embedding_dimensions:
            for use_label in ['emo', 'subject', 'none']:
                start_time = time.time()
                cebra_model = CEBRA(
                    model_architecture=model_name,
                    batch_size=batch_size,
                    temperature_mode="auto",
                    learning_rate=0.001,
                    max_iterations=max_iter,
                    time_offsets=offset,
                    output_dimension=d,
                    device="cuda_if_available",
                    verbose=False
                )
                for i in range(de_train.shape[0]):
                    de_train_i = de_train[i].swapaxes(0, 1).reshape(numTime, -1)
                    if i < de_train.shape[0] - 1:
                        if use_label == 'emo':
                            cebra_model.partial_fit(de_train_i, emo_label_train[i])
                        elif use_label == 'subject':
                            cebra_model.partial_fit(de_train_i, subject_label_train[i])
                        else:
                            cebra_model.partial_fit(de_train_i)
                    else:
                        if use_label == 'emo':
                            cebra_model.fit(de_train_i, emo_label_train[i])
                        elif use_label == 'subject':
                            cebra_model.fit(de_train_i, subject_label_train[i])
                        else:
                            cebra_model.fit(de_train_i)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                logging.info(f'Training model {model_name} with d={d} and label={use_label} took {elapsed_time} seconds')
                print(f'Training model {model_name} with d={d} and label={use_label} took {elapsed_time} seconds')

                cebra_model.save(f'models/de_{model_name}_d{d}_i{max_iter}_label{use_label}.model')
                plt.figure()
                cebra.plot_loss(cebra_model)
                plt.savefig(f'figures/loss_plot_{model_name}_d{d}_i{max_iter}_label{use_label}.png')

def main(config):
    models = config['models']
    batch_size = config['batch_size']
    max_iter = config['max_iter']
    numTime = config['numTime']
    split_data_path = config['split_data_path']
    embedding_dimensions = config['embedding_dimensions']

    de_train, _, emo_label_train, _, subject_label_train, _ = \
        extract_de.load_split_data(split_path=split_data_path)
    train_cebra_models(models, de_train, emo_label_train, subject_label_train,
                       numTime, batch_size, max_iter, embedding_dimensions)

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)
