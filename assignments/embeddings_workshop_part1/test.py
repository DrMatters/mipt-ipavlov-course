import skipgram
import torch

# CONSTANTS --------
VOCAB_SIZE = 5000
BATCH_SIZE = 400
EMBEDDINGS_DIM = 50
EPOCH_NUM = 2
WINDOW_SIZE = 3
# ------------------


text = []
with open('./data/text8', 'r') as text8:
    text = text8.read().split()
# text = ['first', 'used', 'against', 'early', 'working', 'class', 'radicals',
#         'including', 'other', 'another', 'why', 'going', 'because', 'inner', 'product']
batcher = skipgram.TransposeTrickBatcher(text, VOCAB_SIZE, batch_size=BATCH_SIZE,
                                         shuffle_batch=True, window_size=WINDOW_SIZE,
                                         drop_stop_words=True)


loss_history = []
corpus_size = len(batcher._corpus_tokens)

model = skipgram.TransposeTrickSkipGram(VOCAB_SIZE, EMBEDDINGS_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

cumulative_loss = 0
for epoch in range(EPOCH_NUM):
        for i, (batch) in enumerate(batcher):

                # Transform tokens from numpy to torch.Tensor
                tensor_batch = torch.from_numpy(batch).type(torch.long)
                # Send tensors to the selected device

                model.zero_grad()
                loss = model(tensor_batch)
                loss.backward()
                optimizer.step()
                cumulative_loss += loss.item()

                # if i % LOGS_PERIOD == 0:
                print(f'loss on {(i * BATCH_SIZE / corpus_size) * 100:.1f}%:' + \
                      f'{(cumulative_loss) :.7f}')
                loss_history.append(loss.data.cpu().item())
                cumulative_loss = 0

