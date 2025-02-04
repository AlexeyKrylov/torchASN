# coding=utf-8

from __future__ import print_function
import argparse
from collections import Counter
from itertools import chain
import torch

# class VocabEntry(object):
#     def __init__(self):
#         self.word_to_id = dict()
#         self.unk_id = 3
#         self.word_to_id['<pad>'] = 0
#         self.word_to_id['<s>'] = 1
#         self.word_to_id['</s>'] = 2
#         self.word_to_id['<unk>'] = 3

#         self.id2word = {v: k for k, v in self.word_to_id.items()}

#     def __getitem__(self, word):
#         return self.word_to_id.get(word, self.unk_id)

#     def __contains__(self, word):
#         return word in self.word_to_id

#     def __setitem__(self, key, value):
#         raise ValueError('vocabulary is readonly')

#     def __len__(self):
#         return len(self.word_to_id)

#     def __repr__(self):
#         return 'Vocabulary[size=%d]' % len(self)

#     def id2word(self, wid):
#         return self.id2word[wid]

#     def add(self, word):
#         if word not in self:
#             wid = self.word2id[word] = len(self)
#             self.id2word[wid] = word
#             return wid
#         else:
#             return self[word]

#     def is_unk(self, word):
#         return word not in self

#     @staticmethod
#     def from_corpus(corpus, size, freq_cutoff=0):
#         vocab_entry = VocabEntry()

#         word_freq = Counter(chain(*corpus))
#         non_singletons = [w for w in word_freq if word_freq[w] > 1]
#         singletons = [w for w in word_freq if word_freq[w] == 1]
#         print('number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq),
#                                                                                        len(non_singletons)))
#         print('singletons: %s' % singletons)

#         top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)[:size]
#         words_not_included = []
#         for word in top_k_words:
#             if len(vocab_entry) < size:
#                 if word_freq[word] >= freq_cutoff:
#                     vocab_entry.add(word)
#                 else:
#                     words_not_included.append(word)

#         print('word types not included: %s' % words_not_included)

#         return vocab_entry


class VocabEntry(object):
    def __init__(self):
        self.word_to_id = dict()
        self.unk_id = 3
        self.word_to_id['<pad>'] = 0
        self.word_to_id['<sos>'] = 1
        self.word_to_id['<eos>'] = 2
        self.word_to_id['<unk>'] = 3

        self.id_to_word = {v: k for k, v in self.word_to_id.items()}

    def __getitem__(self, word):
        return self.word_to_id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word_to_id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word_to_id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def size(self):
        return len(self.word_to_id)

    def get_word(self, wid):
        return self.id_to_word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word_to_id[word] = len(self)
            self.id_to_word[wid] = word
            return wid
        else:
            return self[word]

    def is_unk(self, word):
        return word not in self

    @classmethod
    def from_corpus(cls, corpus, size, freq_cutoff=0):
        vocab_entry = cls()

        word_freq = Counter(chain(*corpus))
        non_singletons = [w for w in word_freq if word_freq[w] > 1]
        singletons = [w for w in word_freq if word_freq[w] == 1]
        # print('number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq),
        #                                                                                len(non_singletons)))
        # print('singletons: %s' % singletons)

        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)[:size]
        words_not_included = []
        for word in top_k_words:
            if len(vocab_entry) < size:
                if word_freq[word] >= freq_cutoff:
                    vocab_entry.add(word)
                else:
                    words_not_included.append(word)

        print(vocab_entry.id_to_word)
        # print('word types not included: %s' % words_not_included)

        return vocab_entry

class PrimitiveVocabEntry(VocabEntry):
    def __init__(self):
        self.word_to_id = dict()
        self.unk_id = 0
        self.word_to_id['<unk>'] = 0

        self.id_to_word = {v: k for k, v in self.word_to_id.items()}

        
class Vocab(object):
    def __init__(self, **kwargs):
        self.entries = []
        for key, item in kwargs.items():
            assert isinstance(item, VocabEntry)
            self.__setattr__(key, item)

            self.entries.append(key)

    def __repr__(self):
        return 'Vocab(%s)' % (', '.join('%s %swords' % (entry, getattr(self, entry)) for entry in self.entries))

class DatasetVocab(object):
    def __init__(self, src_entry, code_entry, primitive_entries):
        self.src_vocab = src_entry
        self.code_vocab = code_entry
        self.primitive_vocabs = primitive_entries

if __name__ == '__main__':
    raise NotImplementedError
