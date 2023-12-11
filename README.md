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

<p align="center">
<img src="[https://github.com/khuss/user_sentiment_analysis/blob/main/results/imgs/sample_reviews.png]" width="400">
</p>

#### 2.1.2 - Data Exploration:

The reviews length distribution resembles a semi-normal distribution with a denser distribution around the mean and exponentially decreasing as the word count gets further away from the mean.

<p align="center">
<img src="[https://github.com/khuss/user_sentiment_analysis/blob/main/results/imgs/stats_data.png]" width="400">
</p>

Also the mean length of a review is approximately 234 words with a minimum and maximum reviews of 10 and 2470 words respectively.

Finally, the dataset is balanced with +/- 10 000 reviews per class, so there's no need to downsample or upsample to deal with imbalanced classes.

### 2.2 - The Encoder: Bag of Words

In order to turn the words into something our machine can understand we will use an encoder.

The encoder used in the notebook is the bag of words encoder, which based on the corpus provided creates a list of words and assigns a numerical value to each based on its frequency.

For the first 2 models in part 3 (Naive-Bayes and Multi-layer Percetron) we will use `output mode = multi-hot` so our model input will be a vector of 1s and 0s with a size equal to the vocabulary size. 

However for the next models we will use `output mode = int`  and set `sequence_length = 128` for our networks, so our inputs will be a list of integers with a size of 128. We will then use an embedding layer to turn the words into vectors and catch inner-word semantics.

### 2.3 - NLP Models

#### 2.3.1 - Naive Bayes 

The first model we will train is the Gaussian Naive Bayes a probabilistic classifier with the assumption that each class follows a normal distribution and that the features, or words in our case are independent, which is a pretty strong claim and the model's major disadvantage as in real life it is hardly case.

However, it is easy to implement and doesn't require much training data easpecially if the assumption of independence holds.

The accuracy of this on the test set was 75.83 % which is not bad compared to a random baseline model with an approximate accuracy of 50.00 % given the 2 available classes. But still we can do better!

#### 2.3.2 Multi-Layer Perceptron

The second model we explore is the Multi-Layer Perceptron, which is our first and simplest neural network. With only a few layers including the text_vectorization input(encoder), 2 Dense layers and 2 droupout -layers useful to reduce overfitting-.

This model is not only simpler to implement and faster to train than the next more "sophisticated" models but also yielded the best accuracy of 88.23 % on the test set after training for 10 epochs.

It really goes to show "Sometimes simpler is better !"

#### 2.3.3 MLP with an embedding layer

For out third model we will use word embeddings, a vector representation of words as discussed in 2.2. The benefits of having word embeddings is to place words on a multi-dimensional latent space to model the similarities and other relationships between each word.

Besides this layer, the rest of the model is very similar to the Multi-layer Perceptron with 2 Dense layers following the flatten (used to unroll the output of the embedding into a one dimensional) and embedding layers.

The test accuracy of this model while still being decent at 81.84 % is worse than the previous model. The model clearly suffers from overfitting as the training accuracy is 100 %, we can thus see the benefits of having a dropout layer as in the previous model.

#### 2.3.4 LSTM

Next we use LSTMs a type of recurrent neural networks that stands for long short term memory useful when dealing with sequences and very powerful with NLP problems.

While the models stills suffers from overfitting it still presents an improvement in performance with a peak test accuracy of 84.46 %. Improvements, if needed can still be made using dropout layers and other reguralization techniques.

#### 2.3.4 Transformers

Finally, we have the transformer, the state of the art in NLP and language generation. Although they can outperform any model at almost any NLP problem on the long run, as we will come to see, they are limited due to their size and complexity as they require large amounts of computational resources and training time to be effective.

For the first transformer model without positional embeddings, we use the `TransformerEncoder` class i.e the backbone of the model defined in the python file.

For the second transformer model we add the `PositionalEmbedding` class/layer used to equip the overall word embedding representation with its positional information.

After training for 10 epochs the peak test accuracy for the first simple transformer was 84.32 % and 84.66 %.

## 3 - Training results

In the graph below we can see the results of the training session of all models over 10 epochs

<p align="center">
<img src="[https://github.com/khuss/user_sentiment_analysis/blob/main/results/compare_models.png]" width="400">
</p>

## 4 - Conclusions and improvements

After testing multiple models, the multi-layer perceptron presented the highest accuracy of approximately 89 % on the test dataset. Thus it is safe to say that this model can be deployed in a production environment. 
Additionally this model will be cheaper and easier to use in production.

When evaluating the performance of the rest of the models although the Naive Bayes classifier seemed to be underffiting, the rest of the models all suffered from overfitting. And while some can be fixed easily with simple droupout layers  for others it doesn't seem that straightforward.

At the end of the day choosing the right model to deploy in production depends on the maximum accuracy you can get with the lower cost and the constraints on each of these metrics.



