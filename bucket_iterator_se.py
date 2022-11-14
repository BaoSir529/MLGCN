# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy


class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True, max_len = 85):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size, max_len)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size, max_len):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            # print(i)
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size], max_len))
        return batches

    def pad_data(self, batch_data, sentence_max_len):
        batch_text_indices = []
        # batch_context_indices = []
        batch_aspect_indices = []
        batch_pos_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        asp_post = []
        batch_word_height = []
        # batch_dependency_tree = []
        # max_len = max([len(t[self.sort_key]) for t in batch_data])
        max_len = sentence_max_len
        aspect_max_len = max([len(t['aspect_indices']) for t in batch_data])
        for item in batch_data:
            text_indices, aspect_indices, pos_indices, left_indices, polarity, dependency_graph, word_height = \
                item['text_indices'], item['aspect_indices'], item['pos_indices'], item['left_indices'], \
                item['polarity'], item['dependency_graph'], item['word_height']
            # asp_post = torch.sum(left_indices != 0, dim=-1)
            if len(left_indices) > 1 or left_indices[0] != 0:
                asp_post.append(len(left_indices))
            else:
                asp_post.append(0)
            text_padding = [0] * (max_len - len(text_indices))
            aspect_padding = [0] * (aspect_max_len - len(aspect_indices))
            pos_padding = [0] * (max_len - len(pos_indices))
            left_padding = [0] * (max_len - len(left_indices))
            batch_text_indices.append(text_indices + text_padding)
            # batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_pos_indices.append(pos_indices + pos_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_polarity.append(polarity)
            batch_dependency_graph.append(numpy.pad(dependency_graph, \
                ((0, max_len-len(text_indices)), (0, max_len-len(text_indices))), 'constant'))
            # batch_word_height.append(word_height+word_height_padding)
            batch_word_height.append(word_height)
        return { \
                'text_indices': torch.tensor(batch_text_indices), \
                # 'context_indices': torch.tensor(batch_context_indices), \
                'aspect_indices': torch.tensor(batch_aspect_indices), \
                'pos_indices':torch.tensor(batch_pos_indices), \
                'left_indices': torch.tensor(batch_left_indices), \
                'polarity': torch.tensor(batch_polarity), \
                'dependency_graph': torch.tensor(batch_dependency_graph), \
                'asp_post' : torch.tensor(asp_post),\
                'word_height' : batch_word_height,
                # 'dependency_tree': torch.tensor(batch_dependency_tree), \
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
            # pass
        for idx in range(self.batch_len):
            yield self.batches[idx]

