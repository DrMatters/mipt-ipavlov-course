from skipgram import SkipGram, SkipGramBatcher
from torch import nn, optim
import torch

# CONSTANTS --------
VOCAB_SIZE = 5000
BATCH_SIZE = 2
EMBEDDINGS_DIM = 100
EPOCH_NUM = 2
LOGS_PERIOD = 100
# ------------------


text = []
# with open('./data/text8', 'r') as text8:
#     text = text8.read().split()
text = ['first', 'used', 'against', 'early', 'working', 'class', 'radicals', 'including', 'other']
batcher = SkipGramBatcher(text, VOCAB_SIZE, batch_size=BATCH_SIZE, shuffle_batch=True)

for center, context in batcher:
    print(center)
    print(context, '\n')






