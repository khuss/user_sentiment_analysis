# IMDB Reviews: sentiment analysis

In this project we explore different NLP models to build text classifiers to determine the reception for a given movie based on user reviews. 

The classifiers will be trained on the IMDB_reviews dataset available [here]([hshs](https://www.tensorflow.org/datasets/catalog/imdb_reviews)) to determine the overall sentiment of the user reviews, and whether if it is postive or negative.

## 1 - Get Started

Create a conda environment and install all the requirements

`conda create --name sentiment_analysis`

`conda config --append channels conda-forge`

`conda activate sentiment_analysis`

`pip install -r requirements.txt`

## 2 - Python Notebook

### 2.1 - The Dataset

The first thing we need to do after loading the datasets is to have a quick look and do a bit of data exploration to gather some useful statistics for our model design:

#### 2.1.1 - Quick Look:

So the dataset is composed of a text with a label of 1 or 0 if a user liked or disliked the movie respectively.

IMAGE

#### 2.1.2 - Data Exploration:

The reviews length distribution resembles a semi-normal distribution with a denser distribution around the mean and exponentially decreasing as the word count gets further away from the mean.

IMAGE

Also the mean length of a review is approximately 234 words with a minimum and maximum reviews of 10 and 2470 words respectively.

Finally, the dataset is balanced with +/- 10 000 reviews per class, so there's no need to downsample or upsample to deal with imbalanced classes.

### 2.2 - The Encoder: Bag of Words

In order to turn the words into something our machine can understand we will use an encoder.

The encoder used in the notebook is the bag of words encoder, which based on the corpus provided creates a list of words and assigns a numerical value to each based on its frequency.

For the first 2 models in part 3 (Naive-Bayes and Multi-layer Percetron) we will use `output mode = multi-hot` so our model input will be a vector of 1s and 0s with a size equal to the vocabulary size. 

However for the next models we will use `output mode = int`  and set `sequence_length = 128` for our networks, so our inputs will be a list of integers with a size of 128. We will then use an embedding layer to turn the words into vectors and catch inner-word semantics.

### 2.3 - NLP Models

#### 2.3.1 Naive Bayes 

The first model we will train is the Gaussian Naive Bayes a probabilistic classifier with the assumption that each class follows a normal distribution and that the features, or words in our case are independent, (a pretty strong assumption).



