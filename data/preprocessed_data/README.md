# Sample Split Data

## Overview

The file `sample_split_data.npy` is a processed dataset that has been divided and ready for testing our program. It is processed using the script `extract_time.py`.

## Data Structure

### Features

The feature data represents the concatenated original EEG (Electroencephalogram) signals. The shape of this data is defined by two dimensions:

- **Sampling Points**: The number of individual sampling points in the time series data.
- **Channels**: The number of channels used for recording the EEG signals.

This structure (sampling points, channels) illustrates how the raw brainwave data has been organized.

### Labels

The label data corresponds to the individual sampling points, and its shape is defined as:

- **Sampling Points**: This single dimension reflects the label for each sampling point in the data.

It means that every sampling point in the feature data has an associated label, defining the relationship or category it belongs to.

## Training and Testing Sets

The dataset is divided into training and testing subsets:

- **Training Set**: Contains data for 2 subjects, 3 emotions.
- **Testing Set**: Contains data for 1 subject, 3 emotions.

## Usage

To obtain the latent embedding, you can directly run the following command:

```bash
python main.py train config.yaml

