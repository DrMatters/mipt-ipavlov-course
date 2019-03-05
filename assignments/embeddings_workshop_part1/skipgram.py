from torch import nn
from collections import Counter
import gc
import numpy as np
from nltk.corpus import stopwords
import nltk


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.linear_layer = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        embeds = self.embedding_layer(inputs)
        out = self.linear_layer(embeds)
        out = self.activation(out)
        return out

    def get_intrinsic_matrix(self):
        intrinsic = (self.embedding_layer.cpu().weight.data.numpy() +
                     self.linear_layer.cpu().weight.data.numpy())
        return intrinsic


class SkipGramBatcher:
    def __init__(self, corpus, vocab_size, window_size=2,
                 batch_size=3, unk_text='<UNK>', drop_stop_words=True,
                 shuffle_batch=True):
        self.window_size = window_size
        self.vocab_size = vocab_size - 1
        self.batch_size = batch_size
        self.unk_text = unk_text
        self.shuffle_batch = shuffle_batch

        if drop_stop_words:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            cleaned_corpus = [word for word in corpus if not word in stop_words]
            corpus = cleaned_corpus

        # 1. Count all word occurencies.
        self._counted_words = Counter(corpus).most_common(self.vocab_size)
        # create dict using dict comprehension
        self._idx_to_word = {idx: word for idx, (word, count) in enumerate(self._counted_words)}
        self._word_to_idx = {word: idx for idx, (word, count) in enumerate(self._counted_words)}

        # append '<UNK>' token to dictionaries
        last_idx = len(self._idx_to_word)
        self._idx_to_word[last_idx] = self.unk_text
        self._word_to_idx[self.unk_text] = last_idx
        indexed = self.words_to_tokens(corpus, error_on_unk=False)

        # transform corpus from strings to indexes, to reduce memory usage
        self._corpus_indexes = np.asarray(indexed, dtype=np.int32)
        self._batch_shuffled_sequence = np.arange(len(self._corpus_indexes))

        # clean memory
        corpus = []
        gc.collect()

    def words_to_tokens(self, words, error_on_unk=True):
        """Function to transform iterable of words into list of tokens"""

        unk_index = self._word_to_idx[self.unk_text]
        idxes = [self._word_to_idx.get(word, unk_index) for word in words]
        if error_on_unk and unk_index in idxes:
            raise IndexError("Some words are not present in the dictionary")
        return idxes

    def tokens_to_words(self, tokens):
        """Function to transfrom iterable of tokens into list of words"""

        words = [self._idx_to_word[token] for token in tokens]
        return words

    def _get_random_sample(self, center_pos):
        """Internal function to get a random sample within the selected window_size"""

        left_window = np.arange(max(0, center_pos - self.window_size),
                                center_pos)
        right_window = np.arange(center_pos + 1,
                                 min(center_pos + self.window_size + 1, len(self._corpus_indexes)))
        window = np.concatenate((left_window, right_window))
        position = np.random.choice(window)
        return self._corpus_indexes[position]

    def __iter__(self):
        if self.shuffle_batch:
            np.random.shuffle(self._batch_shuffled_sequence)
        self.batch_start_pos = 0
        return self

    def __next__(self):
        if self.batch_start_pos >= len(self._corpus_indexes):
            raise StopIteration
        else:
            batch_position_in_corpus = self._batch_shuffled_sequence[np.arange(
                self.batch_start_pos,
                min(self.batch_start_pos + self.batch_size, len(self._batch_shuffled_sequence))
            )]
            x_batch = np.asarray(self._corpus_indexes[batch_position_in_corpus])
            # draw a word from window of a selected word
            y_batch = np.asarray([self._get_random_sample(selected_word_position)
                                  for selected_word_position in batch_position_in_corpus]).flatten()
            self.batch_start_pos += self.batch_size
            return x_batch, y_batch
