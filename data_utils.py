# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np


def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = './datasets/{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)  # 取名300_rest14_embedding_matrix.pkl
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim))
        fname = './glove/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None,pos2idx=None, POS=False):
        if POS:
            if pos2idx is None:    # 做pos分词
                self.pos2idx = {}
                self.idx2pos = {}
                self.idx = 0
                self.pos2idx['<pad>'] = self.idx
                self.idx2pos[self.idx] = '<pad>'
                self.idx += 1
                self.pos2idx['<unk>'] = self.idx
                self.idx2pos[self.idx] = '<unk>'
                self.idx += 1
            else:
                self.pos2idx = pos2idx
                self.idx2pos = {v: k for k, v in pos2idx.items()}
        elif word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
            self.word2idx['<asp>'] = self.idx
            self.idx2word[self.idx] = '<asp>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def fit_on_pos(self,pos):
        pos_tag = pos.split()
        for tag in pos_tag:
            if tag not in self.pos2idx:
                self.pos2idx[tag] = self.idx
                self.idx2pos[self.idx] = tag
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

    def pos_to_sequence(self, pos):
        pos = pos
        pos_tags = pos.split()
        unknownidx = 1
        sequence = [self.pos2idx[w] if w in self.pos2idx else unknownidx for w in pos_tags]
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod  # 定义静态方法: 即便不实例化类对象,也能够直接调用该方法:即ABSADatesetReader.__read_test__(fnames)
    def __read_text__(fnames):
        text = ''
        # punct = [',', '.', '?', '!', ':', ';', '"', "'", '@', '#', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '/', '<', '>', '~', '`']
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        # for p in punct:
        #     text = text.replace(p, '').strip()
        return text  # 将数据集中的话处理成"sentence_1 sentence_2 ... sentence_n"的一个字符串

    @staticmethod
    def __read_data__(fname, tokenizer_word, tokenizer_pos, tree=False):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        if tree:
            fin = open(fname + '_merger.tree', 'rb')
        else:
            fin = open(fname + '_merger.graph', 'rb')
        idx2graph = pickle.load(fin)

        fin.close()
        fin = open(fname + '.word_height', 'rb')
        height = pickle.load(fin)
        fin.close()

        fin = open(fname + '.pos', 'rb')
        pos = pickle.load(fin)
        pos = pos[1]
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            sentence = lines[i]
            text_left, _, text_right = [s.lower().strip() for s in sentence.partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            # 这里先按照'<asp>'放进去,后面在提取出来单独embedding就是了
            text_indices = tokenizer_word.text_to_sequence(text_left + " " + '<asp>' + " " + text_right)  # 将一句话用idx表示出来
            pos_indices = tokenizer_pos.pos_to_sequence(pos[int(i//3)])
            context_indices = tokenizer_word.text_to_sequence(text_left + " " + text_right)
            aspect_indices = tokenizer_word.text_to_sequence(aspect)
            left_indices = tokenizer_word.text_to_sequence(text_left)
            polarity = int(polarity) + 1  # Neg/Nor/Pos == 0/1/2
            dependency_graph = idx2graph[i]  # 拿到第i句话的依赖图
            # dependency_tree = idx2tree[i]
            word_height = height[i]

            data = {
                'text_indices': text_indices,
                'context_indices': context_indices,
                'aspect_indices': aspect_indices,
                'pos_indices':pos_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                # 'dependency_tree': dependency_tree,
                'word_height': word_height,

            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', embed_dim=300, opt=None):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            },
            'ceshi': {
                'train': './datasets/ceshi/ceshi_train.raw',
                'test': './datasets/ceshi/ceshi_test.raw'
            },

        }
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])  # [train_path , test_path]
        with open(fname[dataset]['train'] + '.pos', 'rb') as f:
            pos = pickle.load(f)
        pos = pos[0]
        if os.path.exists(fname[dataset]['train'][:-10] + '_word2idx.pkl'):  # 加载分词器
            print("loading {0} tokenizer...".format(dataset))
            with open(fname[dataset]['train'][:-10] + '_word2idx.pkl', 'rb') as f:
                word2idx = pickle.load(f)  # 这是个字典,{'but':0, 'the':1,.....}
                tokenizer_word = Tokenizer(word2idx=word2idx)  # tokernizer 就是存储了两个字典的类, word2idx 和 idx2word
        else:
            tokenizer_word = Tokenizer()
            tokenizer_word.fit_on_text(text)
            with open(fname[dataset]['train'][:-10] + '_word2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer_word.word2idx, f)

        if os.path.exists(fname[dataset]['train'][:-10] + '_pos2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(fname[dataset]['train'][:-10] + '_pos2idx.pkl', 'rb') as f:
                pos2idx = pickle.load(f)
                tokenizer_pos = Tokenizer(pos2idx=pos2idx, POS=True)
        else:
            tokenizer_pos = Tokenizer(POS=True)
            tokenizer_pos.fit_on_pos(pos)
            with open(fname[dataset]['train'][:-10] + '_pos2idx.pkl', 'wb') as f:
                pickle.dump(tokenizer_pos.pos2idx, f)
        opt.pos_size = len(tokenizer_pos.idx2pos)
        self.embedding_matrix = build_embedding_matrix(tokenizer_word.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer_word, tokenizer_pos, tree=opt.tree))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer_word, tokenizer_pos))
        # 上面这个ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer)干不少事:
        # 返回一个列表:[{###}, {###},..., {###}]
        # 每个字典存储一句话的详情信息, 包括:
        '''
        {
        'text_indices': 整句话的word2index表示(list),
        'context_indices': 整句话(剔除方面词)的word2index表示(list),
        'aspect_indices': 方面词的word2index表示(list),
        'left_indices': 方面词左边内容的word2index表示(list),
        'polarity': 情感极性0/1/2(int),
        'dependency_graph': 依赖图(ndarray),
        'dependency_tree': 依赖树(ndarray),
        }
        '''
        # 这个列表被定义为一个类ABSADastaset()是为了后续方便调用类下的方法, 很多地方定义类也是这样, 方便程序的统一管理和调度;
        # 以前总感觉定义类有点多此一举, 实际上不然, 通过定义类, 可以将同一类的方法打包写在一起, 非常有利于这种大型工程的模块化管理;
