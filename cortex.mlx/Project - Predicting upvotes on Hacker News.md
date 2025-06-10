## Project Overview

In this project, we will try to predict the upvote score of posts on the _Hacker News_ website [https://news.ycombinator.com/](https://news.ycombinator.com/) using at least their titles.

You are allowed (and recommended) to use _PyTorch_ and its libraries.

This is the recipe which we suggest.

1. **Prepare the dataset of Hacker News titles and upvote scores**  
   - Obtain the data from the database by connecting to it with this connection string:  
     - `postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki`  
     - Use `psql` or some other tool to connect.  
   - Tokenise the titles (see the next chapter)

2. **Implement and train an architecture to obtain word embeddings in the style of the _word2vec_ paper** ([https://arxiv.org/pdf/1301.3781.pdf](https://arxiv.org/pdf/1301.3781.pdf)) using either the _continuous bag of words (CBOW)_ or _Skip-gram_ model (or both).  
   
   We recommend training these on the _text8_ dataset, which you can find in a convenient format below:  
   - [https://huggingface.co/datasets/ardMLX/text8](https://huggingface.co/datasets/ardMLX/text8)  
   
   This is a file of cleaned-up Wikipedia articles by Matt Mahoney; learn more here:  
   - [https://mattmahoney.net/dc/textdata.html](https://mattmahoney.net/dc/textdata.html)

3. **Implement a regression model to predict a Hacker News upvote score from the pooled average of the word embeddings in each title.**

4. **_Extension_**: train your word embeddings on a different dataset, such as:  
   - More Hacker News content, such as comments  
   - A completely different corpus of text, like (some of) Wikipedia

In the following lessons of this week's Cortex module, we will delve into these steps and the technologies which they use.


## Tokenisation

In order to be able to process natural language, such as _Hacker News_ titles, we first need to perform a process called **tokenisation**, which has two main steps.

1. Breaking up the text into a list of atomic units called **tokens**, like words or _sub-words_ (fragments of words). This may involve some processing of the text, like:
   - setting all letters to lowercase  
   - **stemming** and **lemmatisation**, which mean reducing words to their stem (e.g. “running” → “run”) or root respectively  
   - replacing punctuation with special tokens, like ``<COMMA>`` for “,”  
   - removing whitespace  

2. Converting each token to a corresponding **token id**, which is usually a positive integer.

The reason we do this is to be able to transform our text from a string to a list of numbers, which we can feed into our neural network.

## Token Embeddings and Word2Vec

### Token Embeddings

In the last unit, we saw how to break down natural language into _tokens_. Next, we want to turn these tokens into **token embeddings** (sometimes also called _word embeddings_, which can be confusing when we tokenise at a sub-word level). These embeddings are vectors of real numbers corresponding to each token.

There are two main reasons we want to do this:

1. We need a vector representation for our data (in this case, a list of sub-words for each title) so that we can feed it into a neural network.  
2. We want our embeddings to capture the semantic and syntactic content of our tokens. To do this, we assign each token many features by representing them as high-dimensional vectors.

### Word2Vec

One popular way to learn token embeddings comes from the paper _Efficient Estimation of Word Representations in Vector Space_ (a.k.a. **word2vec**):  
https://arxiv.org/pdf/1301.3781.pdf

The **word2vec** recipe is:

1. Tokenise the input text into (sub)word tokens, then convert those tokens into token IDs.  
2. Build a neural network with:  
   - An embedding layer that maps token IDs to embedding vectors (initialized randomly).  
   - A projection + linear output layer that predicts targets for a training task.  
3. Train the network on the chosen task, updating all parameters (including the embeddings).  
4. Extract the trained embedding layer to generate token embeddings for future tasks.

### Training Tasks

The word2vec paper proposes two training tasks (and corresponding architectures):

#### Continuous Bag of Words (CBOW)

- For each token in the text, take _C_ tokens on either side as the **context window**.  
- Given the context window as input, predict the original (center) token.

![CBOW](https://cdn.mlx.institute/assets/CBOW.png)

1. The context tokens are turned into their embeddings and **averaged** in the _Projection Layer_.  
2. The result is linearly transformed and passed through a softmax in the _Output Layer_ to predict the center token.

#### Skip-Gram

- For each token, take _C_ tokens on either side as the **context window**.  
- Given the original (center) token as input, predict each token in the context window.

![Skip-gram](https://cdn.mlx.institute/assets/Skip-gram.png)

1. The center token is turned into its embedding in the _Projection Layer_.  
2. This embedding is linearly transformed and passed through a softmax in the _Output Layer_ to predict each context token.


## Upvote Prediction

Once we have our token embeddings from our word2vec architecture, we can use them for our final task, which is an example of a **regression** task.

To perform this task, we want to implement a neural network which should:

- Take in a Hacker News title  
- Convert it to a list of token embeddings using our word2vec architecture  
- Take the average of those embeddings (this is called **average pooling** and is actually quite a crude technique; we will see how you can do better next week with RNNs)  
- Pass this averaged embedding through a series of hidden layers with widths and activation functions of your choice  
- Pass the result through an output layer (a linear layer with a single neuron) to produce a single number representing the network's prediction for the upvote score  
- Compare the predicted score with the true score (the *label*) via a *Mean Square Error* loss function  

Of course, in reality we will pass our titles and labelled scores in batches, so your network should be written to handle vectorised data in the form of PyTorch *tensors*.

Finally, we will train our neural network on the data we prepared earlier and try to get as accurate a model as possible. *Good luck!*
