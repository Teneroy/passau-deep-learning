# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)

training_data = [
    # Tags are: DET - determiner; NN - noun; V - verb
    # For example, the word "The" is a determiner
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

# TODO: words cannot directly be used as input to a model. Map each unique word in the
# training datato a unique index. Use `word_to_ix` for this purpose.
word_to_ix = {}
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}  # Assign each tag with a unique index
for row in training_data:
    for i in range(len(row[0])):
        word_to_ix[row[0][i]] = tag_to_ix.get(row[1][i])
print(word_to_ix)


# TODO: implement a function that converts a list of strings into a tensor of integers (type
# long) given some dictionary that maps strings to integers
def prepare_sequence(seq: List[str], to_ix: Dict[str, int]):
    return torch.tensor([to_ix.get(elem) for elem in seq], dtype=torch.long)


val = training_data[0][0]

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        """
        Args:
            embedding_dim ([type]): number of dimensions of word embeddings
            hidden_dim ([type]): number of dimensions of hidden state
            vocab_size ([type]): number of unique words
            tagset_size ([type]): number of unique tags (outputs)
        """
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # TODO
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.word_embeddings.embedding_dim, hidden_dim)

        # TODO
        # The linear layer that maps from hidden state space to tag space
        self.lin1 = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # TODO: add the forward pass taking into account the shape that inputs to an LSTM must have (see the start of the notebook)
        # The final output should have shape (len_of_sentence, tagset_size).
        # Depending on the loss function you want to use, you may have to normalize the output
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_scores = self.lin1(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_scores, dim=1)
        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
learning_rate = 1e-3
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
critertion = nn.NLLLoss()

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)


# TODO: train your model for 300 epochs
for epoch in range(300):
    print("Epoch 1")
    for sentence, tags in training_data:
        model.zero_grad()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        tag_scores = model(sentence_in)
        loss = critertion(tag_scores, targets)
        loss.backward()
        optim.step()


# See what the scores are after training
# If your implementation is correct, the model should be able to give the correct tags.
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)
