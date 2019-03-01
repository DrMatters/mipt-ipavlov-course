from torch import nn
from collections import Counter
import gc
import numpy as np


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.linear_layer = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.activation = nn.LogSoftmax(dim=0)

    def forward(self, inputs):
        embeds = self.embedding_layer(inputs)
        out = self.linear_layer(embeds)
        out = self.activation(out)
        return out


class SkipGramBatcher:
    def __init__(self, corpus, vocab_size, window_size=2, batch_size=3, unk_text='<UNK>'):
        self.window_size = window_size
        self.vocab_size = vocab_size - 1
        self.batch_size = batch_size
        self.unk_text = unk_text

        # 1. Count all word occurencies.
        counted_words = Counter(corpus).most_common(self.vocab_size)
        # create dict using dict comprehension
        self.idx_to_word = {idx: word for idx, (word, count) in enumerate(counted_words)}
        self.word_to_idx = {word: idx for idx, (word, count) in enumerate(counted_words)}

        # append '<UNK>' token to dictionaries
        last_idx = len(self.idx_to_word)
        self.idx_to_word[last_idx] = self.unk_text
        self.word_to_idx[self.unk_text] = last_idx
        indexed = self.words_to_indexes(corpus)

        # transform corpus from strings to indexes, to reduce memory usage
        self.corpus_indexes = np.asarray(
            indexed,
            dtype=np.int32
        )

        gc.collect()

    def words_to_indexes(self, words):
        unk_index = self.word_to_idx[self.unk_text]
        idxes = [self.word_to_idx.get(word, unk_index) for word in words]
        return idxes

    def indexes_to_words(self, indexes):
        words = [self.idx_to_word[index] for index in indexes]
        return words

    def __iter__(self):
        self.batch_start_pos = 0
        return self

    def get_random_sample(self, center_id):
        left_window = np.arange(max(0, center_id - self.window_size),
                                center_id)
        right_window = np.arange(center_id + 1,
                                 min(center_id + self.window_size + 1, len(self.corpus_indexes)))
        window = np.concatenate((left_window, right_window))
        position = np.random.choice(window)
        return self.corpus_indexes[position]

    def __next__(self):
        if self.batch_start_pos >= len(self.corpus_indexes):
            raise StopIteration
        else:
            batch_position_in_corpus = np.arange(
                self.batch_start_pos,
                min(self.batch_start_pos + self.batch_size, len(self.corpus_indexes))
            )
            x_batch = np.asarray(self.corpus_indexes[batch_position_in_corpus])
            # draw a word from window of a selected word
            y_batch = np.asarray([self.get_random_sample(selected_word_position)
                                  for selected_word_position in batch_position_in_corpus]).flatten()
            # for selected_word_position in batch_position_in_corpus:
            #     y_batch.append(self.get_random_sample(selected_word_position))
            self.batch_start_pos += self.batch_size
            return x_batch, y_batch
