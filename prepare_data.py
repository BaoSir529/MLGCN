import numpy as np
import spacy
import pickle
from tqdm import tqdm
from spacy.language import Language
from spacy.tokens import Doc
from nltk import Tree


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        doc[token.i + 1].is_sent_start = False
    return doc


nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("set_custom_boundaries", before="parser")
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def find_ancestor(token, matrix):
    for ancestor in token.ancestors:
        matrix[token.i][ancestor.i] = 1
        matrix[ancestor.i][token.i] = 0
        find_ancestor(ancestor, matrix)
        break


def node_high(token, high):
    for ances in token.ancestors:
        if ances is not None:
            high += 1
    return high


def tree_deep(tokens):
    tree = [to_nltk_tree(sent.root) for sent in tokens.sents]
    assert len(tree) == 1
    return tree[0].height()


def dependency_adj_matrix(text, post, tree=False):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    # seq_len = len(text.split())
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))
    # token = document[post]
    for token in tokens:
        matrix[token.i][token.i] = 1
        # https://spacy.io/docs/api/token
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1
        if tree:
            find_ancestor(tokens[post], matrix)
    return matrix


def extract_height(text):
    tokens = nlp(text)
    deep = tree_deep(tokens)
    height = [deep]
    for token in tokens:
        high = node_high(token, 0)
        height.append(high)
    return height


def extract_pos(text):
    tokens = nlp(text)
    words = text.split()
    assert len(words) == len(list(tokens))
    pos_str = list()
    for token in tokens:
        pos_str.append(str(token.pos_))
    return ' '.join(pos_str)


def process(filename, tree=False):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    idx2pos = {0: [], 1: []}
    word_heigth = {}
    if tree:
        fout = open(filename+'_merger.tree', 'wb')
    else:
        fout = open(filename+'_merger.graph', 'wb')
    fout2 = open(filename+'.pos', 'wb')
    fout3 = open(filename+'.word_height', 'wb')
    for i in tqdm(range(0, (len(lines)-1), 3), ):
        sentence = lines[i]
        text_left, _, text_right = [s.lower().strip() for s in sentence.partition("$T$")]
        aspect = 'aspect'
        asp_post = sentence.strip().split().index('$T$')
        graph_undirect = dependency_adj_matrix(text_left + ' ' + aspect + ' ' + text_right, asp_post, tree=tree)
        height = extract_height(text_left + ' ' + aspect + ' ' + text_right)
        sentence_pos = extract_pos(text_left + ' ' + aspect + ' ' + text_right)
        idx2graph[i] = graph_undirect
        word_heigth[i] = height
        idx2pos[0].append(sentence_pos)
        idx2pos[1].append(sentence_pos)
    idx2pos[0] = str(' '.join(idx2pos[0]))
    pickle.dump(idx2graph, fout)
    pickle.dump(idx2pos, fout2)
    pickle.dump(word_heigth, fout3)
    fout.close()
    fout2.close()
    fout3.close()


if __name__ == '__main__':
    for tree in [True, False]:
        process('./datasets/acl-14-short-data/train.raw', tree=tree)
        process('./datasets/acl-14-short-data/test.raw', tree=tree)
        process('./datasets/semeval14/restaurant_train.raw', tree=tree)
        process('./datasets/semeval14/restaurant_test.raw', tree=tree)
        process('./datasets/semeval14/laptop_train.raw', tree=tree)
        process('./datasets/semeval14/laptop_test.raw', tree=tree)
        process('./datasets/semeval15/restaurant_train.raw', tree=tree)
        process('./datasets/semeval15/restaurant_test.raw', tree=tree)
        process('./datasets/semeval16/restaurant_train.raw', tree=tree)
        process('./datasets/semeval16/restaurant_test.raw', tree=tree)
        process('./datasets/ceshi/ceshi_train.raw', tree=tree)
        process('./datasets/ceshi/ceshi_test.raw', tree=tree)
