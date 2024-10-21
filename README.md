# Dynamic Bernoulli Embeddings

This repository is a fork of the original [dynamic_bernoulli_embeddings](https://github.com/llefebure/dynamic_bernoulli_embeddings), enhanced to support PyTorch's Distributed Data Parallel (DDP) for efficient multi-GPU training, enabling faster processing of large datasets, with a focus on training and analyzing models using two key scripts: This repository has been tested with Python 3.9 (Conda), PyTorch 2.5.0, and Ubuntu 22.04.5.

### Install with

To set up the testing environment with conda, use the following instructions:

1. Clone this repository

2. Set up the conda environment:

```

conda config --set channel_priority flexible

conda env create -f environment.yml

```

3. Activate the recreated environment:

```

conda activate DBE_DDP

```

### Key Scripts

1. **training_with_DDP.py**

   This script is used to train the Dynamic Bernoulli Embeddings model. Before training, ensure that you have prepared the required data in the `data` folder:

   - **Data Preparation**: Place a `data.pkl` file inside the `data` folder.
   - **First Run**: The script will generate a dataset pickle file (`dataset.pkl`) during the initial run.
   - **Training**: On the second run, the training process will begin using the prepared dataset.
   - **Model Saving**: After training, the model will be saved to a designated file.

2. **analysis_after_DDP.py**

   This script is used to analyze the trained model:

   - It loads the trained model file and generates embedding data for analysis.
   - The resulting embeddings are saved as `checkpoint/emb.pkl` for further use.

### How to Use

1. **Install Dependencies**

   Make sure to install all necessary dependencies:

2. **Training the Model**

   Run the `training_with_DDP.py` script to prepare and train the model:

3. **Analyzing the Model**

   Once training is complete, use the `analysis_after_DDP.py` script to generate the embedding file:

### Folder Structure

- `checkpoint/`: Stores the trained model files and generated embedding files (`emb.pkl`).
- `data/`: Contains the required dataset (`data.pkl`).
- `dynamic_bernoulli_embeddings/`: Source code for the embeddings model.
