from __future__ import unicode_literals, print_function, division
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

PATH="/home/ubuntu/data/text/names/data/"

from io import open
import glob
import os
import unicodedata
import string
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import random

def findFiles(path): return glob.glob(path)

def uniToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [uniToAscii(line) for line in lines]

for filename in findFiles(PATH+"*.txt"):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

def letterToTensor(letter):
    tensor = torch.zeros(1,n_letters)
    tensor[0][all_letters.find(letter)]=1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line),1,n_letters)
    for li, letter in enumerate(line):
        tensor[li] = letterToTensor(letter)
    return tensor



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.softmax(self.i2o(combined))
        return output, hidden




class RNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN2, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnncell = nn.GRU(input_size, hidden_size, num_layers)
        self.lin = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, input, hidden):
        _, hidden = self.rnncell(input, hidden)
        # take the last step of the output vector
#         output = output[output.size()[0]-1]
        output = lin(hidden.squeeze(dim = 0))
        output = softmax(output[0])
        return output.unsqueeze(0)

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


cat, name, c_tensor, n_tensor = random_sample()
hidden_size = 128
rnn2 = RNN2(n_letters, hidden_size, n_categories, 1)
hidden = rnn2.init_hidden()

output = rnn2(n_tensor, hidden)

print(output)
