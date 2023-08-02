import yaml
from sklearn import svm
from joblib import dump
import time
import logging
import cebra
import numpy as np
import matplotlib.pyplot as plt
import extract_time

def perform_decoding_and_plot(models, de_train, de_test, emo_label_train, emo_label_test, subject_label_train, subject_label_test,
                                max_iter, embeddings, decoder_labels, embedding_dimensions):
    decoder = cebra.KNNDecoder()

    de_reshape_train = de_train.swapaxes(1, 2)
    de_reshape_train = de_reshape_train.reshape(de_reshape_train.shape[0] * de_reshape_train.shape[1], -1)
    de_reshape_test = de_test.swapaxes(1, 2)
    de_reshape_test = de_reshape_test.reshape(de_reshape_test.shape[0] * de_reshape_test.shape[1], -1)

    for model_name, offset in models:
        for d in embedding_dimensions:
            for embedding_type in embeddings:

                model_fullname = f'de_{model_name}_d{d}_i{max_iter}_label{embedding_type}.model'
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
                
                for use_label in decoder_labels:
                    if use_label == 'subject':
                        print('fitting subject decoder')
                        #decoder.fit(embeddings_train, subject_label_train.flatten())
                        decoder.fit(embeddings_train[:1000,:], subject_label_train.flatten()[0:1000])
                        dump(decoder, f'decoders/{model_fullname}_decode_subject_knn.clf')

                        predict_labels_train = decoder.predict(embeddings_train)
                        predict_labels_test = decoder.predict(embeddings_test)
                        #print(predict_labels_train, predict_labels_test)

                        acc_train = np.sum(predict_labels_train == subject_label_train.flatten()) / predict_labels_train.shape[0]
                        acc_test = np.sum(predict_labels_test == subject_label_test.flatten()) / predict_labels_test.shape[0]
                        logging.info(f"Model: {model_fullname}, Decoder_label: {use_label}, Train Accuracy: {acc_train}, Test Accuracy: {acc_test}")
                        print(f"Model: {model_fullname}, Decoder_label: {use_label}, Train Accuracy: {acc_train}, Test Accuracy: {acc_test}")

                    if use_label == 'emo':
                        print('fitting emo decoder')
                        #decoder.fit(embeddings_train, emo_label_train.flatten())
                        decoder.fit(embeddings_train[:1000,:], emo_label_train.flatten()[0:1000])
                        dump(decoder, f'decoders/{model_fullname}_decode_emo_knn.clf')

                        predict_labels_train = decoder.predict(embeddings_train)
                        predict_labels_test = decoder.predict(embeddings_test)
                        # print(predict_labels_train, emo_label_train)
                        # print(predict_labels_test, emo_label_test)

                        acc_train = np.sum(predict_labels_train == emo_label_train.flatten()) / predict_labels_train.shape[0]
                        acc_test = np.sum(predict_labels_test == emo_label_test.flatten()) / predict_labels_test.shape[0]
                        logging.info(f"Model: {model_fullname}, Decoder_label: {use_label}, Train Accuracy: {acc_train}, Test Accuracy: {acc_test}")
                        print(f"Model: {model_fullname}, Decoder_label: {use_label}, Train Accuracy: {acc_train}, Test Accuracy: {acc_test}")

def main(config):

    models = config['models']
    max_iter = config['max_iter']
    split_data_path = config['split_data_path']
    embedding_dimensions = config['embedding_dimensions']
    embeddings = config['embeddings']
    decoder_labels = config['knn_decoder_labels']

    de_train, de_test, emo_label_train, emo_label_test, subject_label_train, subject_label_test = \
        extract_time.load_split_data(split_path=split_data_path)
    perform_decoding_and_plot(models, de_train, de_test, emo_label_train, emo_label_test, subject_label_train, subject_label_test,
                                max_iter, embeddings, decoder_labels, embedding_dimensions)

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    main(config)