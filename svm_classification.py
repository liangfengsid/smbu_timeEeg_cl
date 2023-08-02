import yaml
from sklearn import svm
from joblib import dump
import time
import logging
import cebra
import numpy as np
import matplotlib.pyplot as plt
import extract_time

def perform_decoding_and_plot(models, de_train, de_test, emo_label_train, emo_label_test, 
                                max_iter, embedding_dimensions):
    decoder = svm.NuSVC()

    de_reshape_train = de_train.swapaxes(1, 2)
    de_reshape_train = de_reshape_train.reshape(de_reshape_train.shape[0] * de_reshape_train.shape[1], -1)
    de_reshape_test = de_test.swapaxes(1, 2)
    de_reshape_test = de_reshape_test.reshape(de_reshape_test.shape[0] * de_reshape_test.shape[1], -1)

    for model_name, offset in models:
        for d in embedding_dimensions:
            for use_label in ['none', 'emo']:

                model_fullname = f'de_{model_name}_d{d}_i{max_iter}_label{use_label}.model'
                cebra_model = cebra.CEBRA.load('models/'+model_fullname)

                print(f'transforming data for {model_fullname}')

                embeddings_train = cebra_model.transform(de_reshape_train)
                embeddings_test = cebra_model.transform(de_reshape_test)
                
                plt.figure()
                cebra.plot_embedding(embeddings_train, embedding_labels='time')
                plt.savefig(f'figures/{model_fullname}_embeddings_train.png')

                plt.figure()
                cebra.plot_embedding(embeddings_test, embedding_labels='time')
                plt.savefig(f'figures/{model_fullname}_embeddings_test.png')

                print('figs saved')
                print('fitting decoder')

                decoder.fit(embeddings_train, emo_label_train.flatten())
                #decoder.fit(embeddings_train[:1000,:], emo_label_train.flatten()[0:1000])
                dump(decoder, f'decoders/{model_fullname}_decode_emo_NuSVC.clf')

                predict_labels_train = decoder.predict(embeddings_train)
                predict_labels_test = decoder.predict(embeddings_test)
                acc_train = np.sum(predict_labels_train == emo_label_train.flatten()) / predict_labels_train.shape[0]
                acc_test = np.sum(predict_labels_test == emo_label_test.flatten()) / predict_labels_test.shape[0]

                logging.info(f"Model: {model_fullname}, Training accuracy: {acc_train}, Testing accuracy: {acc_test}")
                print(f"Model: {model_fullname}, Training accuracy: {acc_train}, Testing accuracy: {acc_test}")

def main(config):

    models = config['models']
    max_iter = config['max_iter']
    split_data_path = config['split_data_path']
    embedding_dimensions = config['embedding_dimensions']

    de_train, de_test, emo_label_train, emo_label_test, _, _ = \
        extract_time.load_split_data(split_path=split_data_path)
    perform_decoding_and_plot(models, de_train, de_test, emo_label_train, emo_label_test, 
                                max_iter, embedding_dimensions)

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)