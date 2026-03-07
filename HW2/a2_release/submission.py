import numpy as np
from scipy.special import hankel2e
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

def get_word_embeddings(model, sequence):
    """
    Inputs:
    model: Word2Vec (TODO 1) or KeyedVectors (TODO 4) model.
    sequence: A text string (sequence of words) from which to extract embeddings.

    Outputs:
    word_embeddings: The word embeddings for each word in the input sequence. If a word in the
                     sequence is not included in the word2vec model's vocabulary, that word
                     should be ignored and should not be added to the output word_embeddings.
    """
    word_embeddings = []

    ## TODO 1: implement function for Word2Vec model
    # isinstance(object, type)
    if isinstance(model, Word2Vec):
      for word in sequence.split():
        if word in model.wv:
          word_embeddings.append(model.wv[word])

    ## TODO 4: add support for KeyedVectors model
    #################
    elif isinstance(model, KeyedVectors):
      for word in sequence.split():
        if word in model:
          word_embeddings.append(model[word])

    return word_embeddings


def get_reviews_embeddings(model, data):
    """
    Inputs:
    model: word2vec model(any).
    data: Cleaned dataframe containing n reviews .
          This dataframe has two coloumns: 'text' (the review) and 'label' (whether the review is positive or negative).

    Outputs:
    reviews_embeddings: A list of length n, containing the word embeddings for each
                        review in the input data generated using the input Word2Vec model, i.e.
                        review_embeddings[i] are the word embeddings for the (i + 1)th review
                        in the dataframe.
    """
    reviews_embeddings = []
    ## TODO 2: implement function
    #################
    for review in data['text']:
      word_embedding = get_word_embeddings(model, review)
      reviews_embeddings.append(word_embedding)
    return reviews_embeddings


def max_pool(embeddings, d=300):
    """
    Input:
    embeddings (l x d): The word embeddings of a single review.
    d: The dimension of the embedding.

    Output:
    pooled_embeddings (d): Max-pooled word embeddings.
    """
    max_pool_embeddings = []
    ## TODO 3: max pooling
    #################
    if len(embeddings)==0:
      return [0]*d
    for i in range(d): # iterate all dimensions
      column = [row[i] for row in embeddings] # embedding for dimension i
      max_pool_embeddings.append(max(column))

    return max_pool_embeddings

class LSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, seq_length, num_classes=2):
        """
        Inputs:
        num_layers: Number of recurrent layers
        input_size: Number of features for input
        hidden_size: Number of features in hidden state
        seq_length: Length of sequences in a batch
        num_classes: Number of categories for labels

        Outputs: none
        """
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        ## TODO 5: define LSTM components
        #################
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=hidden_size*num_layers, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)


    def forward(self, x):
        """
        Inputs:
        x: input data

        Outputs:
        out: output of forward pass
        """
        ## TODO 5: implement the forward function based on the architecture described above
        #################
        out, (hn, cn) = self.lstm(x) 
        x = hn.transpose(0, 1) 
        x = x.reshape(x.size(0), -1)  
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def reviews_processing(google_embeddings, length):
    """
    This function takes in a dataset of embeddings and a length to clip these sequences to

    Inputs:
    google_embeddings: a dataset of n reviews that have been embedded with
                       google's w2v with embedding size d
    length: the length you are making sure all sequences are capped at

    Output:
    embeddings (n x length x d): a numpy array representing the dataset where all
                                 reviews are clipped to the input length. If the sequence
                                 is shorter than the specified length, pad the sequence
                                 with zeros.
    """
    embeddings = []
    ## TODO 6: Implement reviews_processing to modify the embeddings
    ###########
    d = len(google_embeddings[0][0]) # embedding size for vocabulary embeding
    for embedding in google_embeddings:
      if len(embedding)>length:
        embeddings.append(embedding[:length])
      else:
        padding_size = length-len(embedding)
        padding = np.zeros((padding_size, d), dtype=np.float32) # zero matrix and its size is (padding_size, d)
        if len(embedding)==0:
          embeddings.append(padding)
        else:
          new_emb = np.vstack((embedding, padding))
          embeddings.append(new_emb)
    return np.array(embeddings, dtype=np.float32)


def val(model, val_loader, criterion, device):
    """
    Inputs:
    model (torch.nn.Module): The deep learning model to be trained.
    val_data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    criterion (torch.nn.Module): Loss function to compute the training loss.

    Outputs:
    Tuple of (validation loss, validation accuracy)
    """
    val_running_loss = 0
    num_correct = 0
    total = 0

    model.eval() # dropout, batch normalization
    with torch.no_grad(): # close the computation of gradients
        for i, (inputs, labels) in enumerate(val_loader, 0): # iterate the batch in val_loader

            # TODO 7: write validation loop body
            #################
              
            inputs = inputs.to(device)
            labels = labels.to(device)            
            outputs = model(inputs)
            # output.shape = (batch_size, num_classes)
            loss = criterion(outputs, labels)
            val_running_loss+=loss.item()

            _, preds = torch.max(outputs, 1) # find the maximal value on dimension 1 (which is num_classes)
            num_correct+=(preds==labels).sum() # .item() change the tensor into python
            total+=labels.size(0)
    model.train() # train model
    return val_running_loss, (num_correct / total).item()

def train(model, train_loader, val_loader, criterion, epochs, optimizer, device):
    """
    Inputs:
    model (torch.nn.Module): The deep learning model to be trained.
    train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    criterion (torch.nn.Module): Loss function to compute the training loss.
    epochs: Number of epochs to train for.
    optimizer: The optimizer to use during training.

    Outputs:
    Tuple of (train_loss_arr, val_loss_arr, val_acc_arr)
    """
    train_loss_arr = []
    val_loss_arr = []
    val_acc_arr = []
    running_loss = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader): # iterate all the batch in train_loader
            # TODO 7: write train loop body
            #################
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() # update parameter by using gradients and learning rate
            running_loss+=loss.item()

        val_loss, val_acc = val(model, val_loader, criterion, device)
        train_loss_arr.append(running_loss)
        val_loss_arr.append(val_loss)
        val_acc_arr.append(val_acc)
        print(
            "epoch:",
            epoch + 1,
            "training loss:",
            running_loss,
            "val accuracy:",
            round(val_acc, 3),
        )

    print(running_loss)
    print("Training finished.")

    return train_loss_arr, val_loss_arr, val_acc_arr


# TODO 8: Understand this code.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        max_seq_length: Maximum length of sequences input into the transformer.
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).reshape(max_seq_length, 1)
        div_term = torch.exp( 
              -1 * (torch.arange(0, d_model, 2).float()/d_model) * math.log(10000.0)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Adds the positional encoding to the model input x.
        """
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: The number of attention heads to use.
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO 9.1: define layers W_q, W_k, W_v, and W_o
        # Hint: Recall that linear layers essentially perform matrix multiplication
        #       between the layer input and layer weights
        #################
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)


    def split_heads(self, x):
        """
        Reshapes Q, K, V into multiple heads.
        """
        batch_size, seq_length, d_model = x.size()
        # [batch_size, num_heads, seq_length, d_model]
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)

    def compute_attention(self, Q, K, V):
        """
        Returns single-headed attention between Q, K, and V.
        """
        # TODO 9.2: compute attention using the attention equation provided above
        #################
        scores = Q@K.transpose(-2, -1)/math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attention = weights@V
        return attention

    def combine_heads(self, x):
        """
        Concatenates the outputs of each attention head into a single output.
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, x):
        # TODO: 9.3 implement forward pass
        #################
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention = self.compute_attention(Q, K, V)

        attn = self.combine_heads(attention)

        output = self.W_o(attn)
        return output



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        d_ff: Hidden dimension size for the feed-forward network.
        """
        super(FeedForward, self).__init__()

        # TODO 10: define the network
        #################
        self.fc1 = nn.Linear(in_features = d_model, out_features = d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features = d_ff, out_features = d_model)

    def forward(self, x):
        # TODO 10: implement feed forward pass
        #################
        x = self.fc2(self.relu(self.fc1(x)))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        d_ff: Hidden dimension size for the feed-forward network.
        p: Dropout probability.
        """
        super(EncoderLayer, self).__init__()

        # TODO 11: define the encoder layer
        #################
        self.p = p
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(p)


    def forward(self, x):

        ## TODO 11: implement the forward function based on the architecture described above
        #################
        # ---- self attention block
        attn_out = self.self_attn(x)
        x = self.norm1(x+self.dropout(attn_out))

        # ---- feed forward block
        ff_out = self.feed_forward(x)
        x = self.norm2(x+self.dropout(ff_out))
        return x


class Transformer(nn.Module):
    def __init__(
        self, num_classes, d_model, num_heads, num_layers, d_ff, max_seq_length, p
    ):
        """
        Inputs:
        num_classes: Number of classes in the classification output.
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        num_layers: Number of encoder layers.
        d_ff: Hidden dimension size for the feed-forward network.
        max_seq_length: Maximum sequence length accepted by the transformer.
        p: Dropout probability.
        """
        super(Transformer, self).__init__()

        # TODO 12: define the transformer
        #################
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(p)
        self.encoder_layers = nn.ModuleList([
          EncoderLayer(d_model, num_heads, d_ff, p)
          for _ in range(num_layers)
        ])
        self.fc1 = nn.Linear(in_features = d_model, out_features = 128)
        self.fc2 = nn.Linear(in_features = 128, out_features = num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):

        ## TODO 12: implement the forward pass
        #################
        # --- positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # --- stacked encode layers
        for layer in self.encoder_layers:
          x = layer(x)

        # --- mean pool
        x = x.mean(dim=1) # (batch_size, seq_len, d_model) ==> (batch_size, d_model)

        # --- classification head
        x = self.fc1(self.relu(x))
        x = self.fc2(self.relu(x))

        return x


def process_batch(bert_model, data, criterion, device, val=False):
    """
    Inputs:
    data: The data in the batch to process.
    criterion: The loss function.
    val: True if processing a batch from the validation or test set.
         False if processing a batching from the training set.

    Outputs:
    Tuple of (outputs, losses)
        outputs: a dictionary containing the model outputs ('out') and predicted labels ('preds')
        metrics: a dictionary containing the model loss over the batch ('loss') and during validation (val = True),
                 the total number of examples in the batch ('batch_size') and the total number of examples whose
                 label the model predicted correctly ('num_correct')
    """

    outputs, metrics = dict(), dict()

    # TODO 13: process batch
    # Hint: For details on what information the data from the data loader contains
    #       check the __getitem__ function defined in the CustomClassDataset implemented
    #       at the beginning of Part 5
    # Hint: Make sure to send the data to the same device that the model is on.
    input_ids = data["source_ids"].to(device)
    att_mask = data["source_mask"].to(device)
    labels = data["label"].to(device).long()
    model_out = bert_model(input_ids=input_ids, attention_mask=att_mask)
    logits = model_out.logits if hasattr(model_out, "logits") else model_out[0]

    loss = criterion(logits, labels)
    preds = torch.argmax(logits, dim=-1)
    metrics["loss"] = loss
    outputs["out"] = logits
    outputs["preds"] = preds

    if val:
        metrics["batch_size"] = torch.tensor(labels.size(0), device=device)
        metrics["num_correct"] = torch.eq(preds, labels).sum()
    #################

    return outputs, metrics
