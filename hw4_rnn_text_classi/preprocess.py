import torch
from torch import nn
from gensim.models import Word2Vec

class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path="./w2v.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def load_w2v_model(self):
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        # add new word to embedding system，give a randomly generate representation vector
        # word is "<PAD>" or "<UNK>"
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        print("Get embedding ...")
        if load:
            print("loading word to vec model ...")
            self.load_w2v_model()
        else:
            raise NotImplementedError

        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i+1), end='\r')
            #make a word to index dictionary e.g. self.word2index['he'] = 1 
            self.word2idx[word] = len(self.word2idx)
            #make a index to word list e.g. self.index2word[1] = 'he'
            self.idx2word.append(word)
            #make a index to word vector list e.g. self.embedding[1] = 'he' vector (250dim)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        print("word",word)
        print("index",i)
        print("vector length",len(self.embedding[word]))

        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format((self.embedding_matrix).shape))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # padding every sentance to the same sen_len by trim and pad
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
                #last_word = sentence[-1]
                #sentence.append(last_word)
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        # transform words in sentence to indexes
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)
