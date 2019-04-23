# Anomaly Detection using Prednet
This project consists of the following steps:

   1.Pre-processing data and generating hickle files using process.py to easily run train and inference <br />
   2.Setting the required hyperparameters and feeding the dataset to train a Prednet model from scratch on the data with train.py <br />
   3.Running inference with the trained model on the test dataset using evaluate.py and collecting various metrics and visualizations to analyse the performance <br />

Processing data: <br />
   The UCSD Anomaly Detection Dataset (http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm), needs to be downloaded and extracted before we can move further.<br />
   Do note that we had randomly split the test folder of both Ped1 and Ped2, to get some validation data. And we did some pre-processing to shift the ground-truth folders in the Test folder to a separate GT folder so that we dont end up processing it by mistake. <br />
   
   The following files are used:<br />
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ii) process.py - This code will take path to the dataset and load each frame of each video, preprocess it and write it to the hickle file. <br />
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                This automatically separates files for train, test and validation data. <br />

Once the above step is completed, the we will be ready to train our models.<br />

The training process:<br />
   The following files are used:<br />
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i) train.py - This file instantiates the Prednet model, the loss function and the optimizer.<br />
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; It then trains the model for the choice of hyperparameters that are set by us using the fix_hyperparams function. <br />
               &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  We document the training logs, checkpoint the model to save the best one that can be used later in the inference process.<br />
			   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  To learn more about Prednet, visit https://coxlab.github.io/prednet/ which has both the paper and the code in more detail. <br />
        
Finally we will be ready to run inference and analyse our performance:<br />
   The following files are used:<br />
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i) train.py - This file instantiates the Prednet model, the loss function and the optimizer.<br />
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; It then trains the model for the choice of hyperparameters that are set by us using the fix_hyperparams function. <br />
               &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  We document the training logs, checkpoint the model to save the best one that can be used later in the inference process.<br />
			   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  To learn more about Prednet, visit https://coxlab.github.io/prednet/ which has both the paper and the code in more detail. <br />


## Libraries used
nltk, Beautiful soup, collections, numpy, itertools, string, re, os, matplotlib.pyplot, math, pickle, lxml, json, stemming, operator

##Development/IDEs
Eclipse(4.0 or above)

## Installation
 
This project requires Python 3.5 and Java 8.0. It also requires Lucene 8.0 which can be downloaded from <https://lucene.apache.org/core/downloads.html>.
There are a few python packages which have to be installed to be able to run the code. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install them as follows:

```bash
pip install beautifulsoup4
pip install nltk
pip install collections
pip install beautifulsoup4
pip install nltk
pip install collections
pip install gensim
pip install pickle
pip install csv

```
## Usage

```bash
python corpus_generator.py 
python unigram_indexer.py
python relevance.py 
python generate_stemmed_corpus.py
python BM25.py
python pseudorelevance_feedback.py
python thesaurusExpansion.py
python tfidf.py
python QMD.py
python evaluation.py
```
The following steps have to be followed for snippet generation:

## Usage

```bash
python snippetGenerationWithHL.py
python resultHTMLBuilder.py
```

For Lucene:

Run IndexFiles.java (after modifying paths) followed by SearchFiles.java in Eclipse. Enter the query term to retrieve the top 100 documents.
