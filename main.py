import sys
import yaml
import logging
import extract_time
import train_models
import classification

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <command> <config_path>")
        sys.exit(1)

    command = sys.argv[1]
    config_path = sys.argv[2]

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if command == 'preprocess':
        extract_time.main(config)
    elif command == 'train':
        logging.basicConfig(filename='logs/training.log', filemode='w', level=logging.INFO,
                format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        train_models.main(config)
    elif command == 'svm_decode':
        logging.basicConfig(filename='logs/svm.log', filemode='w', level=logging.INFO,
                format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        classification.main(config, decoder_type='svm')
    elif command == 'knn_decode':
        logging.basicConfig(filename='logs/knn.log', filemode='w', level=logging.INFO,
                format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        classification.main(config, decoder_type='knn')
    else:
        print("Unknown command:", command)

if __name__ == "__main__":
    main()
