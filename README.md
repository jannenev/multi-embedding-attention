# multi-embedding-attention
Multi-channel embeddings with Attention

solution to 
#### Quora Insincere Questions Classification
https://www.kaggle.com/c/quora-insincere-questions-classification

#### Multiple embeddings are used for same text 
(Glove 300-dim, Fasttext 300-dim, Paragram 300-dim)

Each embedded text is fed into LSTM, extracted Max Pool and Average Pool, Plus LSTM is fed to Attention.
Then these 3 outputs from each embeddig-line are concatenated together and fed to one more Attention layer, followed by Dense layer.

Example of one Embedding line
```
Text-> Glove-300 -> Dropout -> CuDNNLSTM -> Dropout -> max_pool -> concat -> Attention - Dense 
                                         -> Dropout -> avg_pool
                                                    -> Attention
```
                                          

#### Stratified K-fold
For regularization Stratified K-fold is used. For example data is split into 5 random parts, from which 4 of 5 are chosen for each model trained.
Then predictions of all models are combined at the end.

#### Pseudo Labeling
Pseudo Labeling is also applied.
First a smaller and faster (because of time constraint) LSTM64 with 2 embedding lines is used to train 2 models, which are averaged over predictions.
Then training-set is predicted and the estimated highest accuray items are assigned predicted labels to from Pseudo-labeled set, which is add to the original training set and used for training the Main model.

This Pseudo Learning part doesn't yet have optimization to find out what is the optimal threshold - how many percentage of highest accuracy predictions for 0 and 1 to keep.


#### Focal loss
For loss function is used Focal loss, which compard to binary-crossentropy can give lower weight for easy examples and higher weight for hard examples.
