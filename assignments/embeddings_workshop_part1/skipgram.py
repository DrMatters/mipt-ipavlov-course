from torch import nn
import torch
import collections
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


class NegativeSamplingSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(NegativeSamplingSkipGram, self).__init__()

        self.input_emb = nn.Embedding(vocab_size, embedding_dim)
        self.output_emb = nn.Embedding(vocab_size, embedding_dim)
        self.activation = nn.LogSigmoid()

        # init
        torch.nn.init.xavier_uniform_(self.input_emb.weight)
        torch.nn.init.xavier_uniform_(self.output_emb.weight)

    def forward(self, target, context, negative_word_batch):
        """Forward propagate the model"""

        # u,v: [batch_size, emb_dim]
        v = self.input_emb(target)
        u = self.output_emb(context)

        # positive_val: [batch_size]
        positive_val = self.activation(torch.sum(u * v, dim=1)).squeeze()

        # u_hat: [batch_size, neg_size, emb_dim]
        u_hat = self.output_emb(negative_word_batch)

        # [batch_size, neg_size, emb_dim] x [batch_size, emb_dim, 1] = [batch_size, neg_size, 1]
        # neg_vals: [batch_size, neg_size]
        neg_vals = torch.bmm(u_hat, v.unsqueeze(2)).squeeze(2)

        # neg_val: [batch_size]
        neg_val = self.activation(-torch.sum(neg_vals, dim=1)).squeeze()

        loss = positive_val + neg_val
        return -loss.mean()

    def get_intrinsic_matrix(self):
        intrinsic = (self.input_emb.cpu().weight.data.numpy() +
                     self.output_emb.cpu().weight.data.numpy())
        return intrinsic


class SingleMatrixSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device):
        torch.manual_seed(42)
        super(SingleMatrixSkipGram, self).__init__()

        self.emb_matrix = torch.randn(vocab_size, embedding_dim, requires_grad=True, device=device)
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        """Forward propagate the single matrix model

        inputs - takes a list of word tokens
        """
        selected = self.emb_matrix[inputs, :]
        out = torch.mm(selected, torch.transpose(self.emb_matrix, 0, 1))
        out = self.activation(out)
        return out

    def get_intrinsic_matrix(self):
        return self.emb_matrix.cpu().detach().numpy()


class TransposeTrickSkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        torch.manual_seed(42)
        super(TransposeTrickSkipGram, self).__init__()

        self.emb = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, batch):
        """Forward propagate the model"""
        S = self.emb(batch)
        S = nn.functional.normalize(S, dim=2)
        x = torch.sum(
            torch.mean(
                torch.bmm(S, torch.transpose(S, 1, 2)) - torch.cuda.FloatTensor(1).fill_(1), (1, 2)
            )
        )

        # To get a matrix with size [2 * window_size + 1; batch_size; batch_size]
        # we use BMM, which performs a batch matrix-matrix product of matrices stored in batch1 and batch2
        # and handles tensor sizes this way: (b * n * m) @ (b * m * p) -> (b * n * p).
        # Elem1 will have the shape:
        # [batch_size; 2 * window_size + 1; embedding_size] -> [2 * window_size + 1; batch_size; embedding_size]
        # Elem2 will have the shape:
        # [batch_size; 2 * window_size + 1; embedding_size] -> [2 * window_size + 1; batch_size; embedding_size] ->
        # [2 * window_size + 1; embedding_size; batch_size]
        # Thus, after applying BMM we will have desired dimensions
        elem1 = torch.transpose(S, 0, 1)
        elem2 = torch.transpose(S, 0, 1)
        elem2 = torch.transpose(elem2, 1, 2)
        y = torch.sum(
            torch.mean(
                torch.bmm(elem1, elem2), (1, 2)
            )
        )

        loss = -x + y
        return loss
    
    def get_intrinsic_matrix(self):
        intrinsic = self.emb.cpu().weight.data.numpy()
        return intrinsic


class SkipGramBatcher:
    def __init__(self, corpus, vocab_size, window_size=3,
                 batch_size=128, drop_stop_words=True,
                 shuffle_batch=True, unk_text='<UNK>'):
        self.window_size = window_size
        self.vocab_size = vocab_size - 1
        self.batch_size = batch_size
        self.unk_text = unk_text
        self.shuffle_batch = shuffle_batch

        # drop stop words from corpus if it's needed
        if drop_stop_words:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            cleaned_corpus = [word for word in corpus if not word in stop_words]
            corpus = cleaned_corpus

        # Count all word occurrences and select vocab_size most common
        self._counted_words = collections.Counter(corpus).most_common(self.vocab_size)
        # create mappings using dict comprehension
        self._token_to_word = {idx: word for idx, (word, count) in enumerate(self._counted_words)}
        self._word_to_token = {word: idx for idx, (word, count) in enumerate(self._counted_words)}

        # append '<UNK>' token to dictionaries
        last_token = len(self._token_to_word)
        self._token_to_word[last_token] = self.unk_text
        self._word_to_token[self.unk_text] = last_token
        tokenized = self.words_to_tokens(corpus, error_on_unk=False)

        # transform corpus from strings to tokens, to reduce memory usage
        self._corpus_tokens = np.asarray(tokenized, dtype=np.int32)

        # create shuffled sequence to make batch sampling random
        self._batch_shuffled_sequence = np.arange(len(self._corpus_tokens))

        # clean memory
        corpus = []
        gc.collect()

    def words_to_tokens(self, words, error_on_unk=True):
        """Transform iterable of words into list of tokens"""

        unk_index = self._word_to_token[self.unk_text]
        idxes = [self._word_to_token.get(word, unk_index) for word in words]
        if error_on_unk and unk_index in idxes:
            raise IndexError("Some words are not present in the dictionary")
        return idxes

    def tokens_to_words(self, tokens):
        """Transform iterable of tokens into list of words"""

        words = [self._token_to_word[token] for token in tokens]
        return words

    def _get_random_positive_sample(self, center_pos):
        """Internal function to get a random sample within the selected window_size"""

        left_window = np.arange(max(0, center_pos - self.window_size),
                                center_pos)
        right_window = np.arange(center_pos + 1,
                                 min(center_pos + self.window_size + 1, len(self._corpus_tokens)))
        window = np.concatenate((left_window, right_window))
        position = np.random.choice(window)
        return self._corpus_tokens[position]

    def __iter__(self):
        if self.shuffle_batch:
            np.random.shuffle(self._batch_shuffled_sequence)
        self.batch_start_pos = 0
        return self

    def __next__(self):
        if self.batch_start_pos >= len(self._corpus_tokens):
            raise StopIteration
        else:
            # get a list of shuffled numbers
            batch_position_in_corpus = self._batch_shuffled_sequence[np.arange(
                self.batch_start_pos,
                min(self.batch_start_pos + self.batch_size, len(self._batch_shuffled_sequence))
            )]
            center_words_batch = self._corpus_tokens[batch_position_in_corpus]
            # draw a word from window of a selected word
            context_words_batch = np.asarray([self._get_random_positive_sample(selected_word_position)
                                              for selected_word_position in batch_position_in_corpus]).flatten()
            self.batch_start_pos += self.batch_size
            return center_words_batch, context_words_batch


class TransposeTrickBatcher(SkipGramBatcher):
    def _get_full_window(self, center_pos):
        """Get a window of words from batch including the center word"""
        window_pos = np.arange(max(0, center_pos - self.window_size),
                               min(center_pos + self.window_size + 1, len(self._corpus_tokens)))
        window = self._corpus_tokens[window_pos]
        return window

    def __iter__(self):
        self._batch_shuffled_sequence = np.arange(self.window_size, len(self._corpus_tokens) - self.window_size)
        if self.shuffle_batch:
            np.random.shuffle(self._batch_shuffled_sequence)
        self.batch_start_pos = 0
        return self

    def __next__(self):
        """Iterate over batches with each call"""
        if self.batch_start_pos < len(self._batch_shuffled_sequence):
            batch_position_in_corpus = self._batch_shuffled_sequence[np.arange(
                self.batch_start_pos,
                min(self.batch_start_pos + self.batch_size, len(self._batch_shuffled_sequence))
            )]
            batch = np.asarray([self._get_full_window(center_pos) for center_pos in batch_position_in_corpus])
            self.batch_start_pos += self.batch_size
            return batch
        else:
            raise StopIteration


class NegativeSamplingBatcher(SkipGramBatcher):
    def __init__(self, corpus, vocab_size, window_size=3,
                 batch_size=128, drop_stop_words=True,
                 shuffle_batch=True, n_negative_examples=5, unk_text='<UNK>'):
        # initialize base class
        super(NegativeSamplingBatcher, self).__init__(corpus, vocab_size, window_size, batch_size,
                                                      drop_stop_words, shuffle_batch, unk_text)
        self._n_negative_examples = n_negative_examples

        # calculate probabilities of drawing a negative example
        frequencies_np = np.asarray(self._counted_words)[:, 1].astype(np.int32)
        unk_freq = np.count_nonzero(self._corpus_tokens == self._word_to_token[self.unk_text])
        frequencies_np = np.append(frequencies_np, unk_freq)
        powered_frequencies = frequencies_np ** (3/4)
        self._probabilities = powered_frequencies / np.sum(powered_frequencies)

    def _get_negative_samples(self):
        """Internal function to get random samples from corpus distribution"""
        size = self._n_negative_examples
        return np.random.choice(len(self._probabilities), p=self._probabilities, size=size)

    def __next__(self):
        center_words_batch, positive_words_batch = super(NegativeSamplingBatcher, self).__next__()
        neg = np.random.choice(len(self._probabilities), p=self._probabilities,
                               size=(self.batch_size, self._n_negative_examples))
        return center_words_batch, positive_words_batch, neg

