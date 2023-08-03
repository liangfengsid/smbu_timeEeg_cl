# Sample Split Data

## Overview

The file `sample_split_data.npy` is a processed dataset that has been divided and ready for testing our program. It is processed using the script `extract_de.py`.

## Data Structure

### Features

The feature data in the dataset consists of several dimensions, defined as follows:

- `time`: Number of records used for recording the signals.
- `channels`: Number of channels used for recording the signals.


### Labels

The label data in the dataset is also organized in dimensions:

- `time`: Number of records' label used for recording the signals.

## Training and Testing Sets

The dataset is divided into training and testing subsets:

- **Training Set**: Contains data for 2 subjects, 3 emotions.
- **Testing Set**: Contains data for 1 subject, 3 emotions.

## Usage

To obtain the latent embedding, you can directly run the following command:

```bash
python main.py train config.yaml

