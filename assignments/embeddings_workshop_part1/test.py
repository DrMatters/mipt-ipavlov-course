from skipgram import NegativeSamplingBatcher
from torch import nn, optim
import torch

# CONSTANTS --------
VOCAB_SIZE = 10
BATCH_SIZE = 2
EMBEDDINGS_DIM = 5
EPOCH_NUM = 1
LOGS_PERIOD = 2
# ------------------


text = []
# with open('./data/text8', 'r') as text8:
#     text = text8.read().split()
text = ['first', 'used', 'against', 'early', 'working', 'class', 'radicals',
        'including', 'other', 'another', 'is', 'going', 'on']
batcher = NegativeSamplingBatcher(text, VOCAB_SIZE, batch_size=BATCH_SIZE, shuffle_batch=False, n_negative_examples=5)

for center_batch, positive_batch, negative_batch in batcher:
        print(f'Center: {center_batch}')
        print(f'Positive: {positive_batch}')
        print(f'Negative:\n{negative_batch}\n')






