import sys
sys.path.extend(["../","./"])

class NMTVocab:
    PAD, BEGIN_SEQ, END_SEQ, UNK = 0, 1, 2, 3
    S_PAD, S_BEGIN_SEQ, S_END_SEQ, S_UNK = '<pad>', '<s>', '</s>', '<unk>'
    def __init__(self, word_list):
        """
        :param word_list: list of words
        """
        self.i2w = [self.S_PAD, self.S_BEGIN_SEQ, self.S_END_SEQ, self.S_UNK] + word_list

        reverse = lambda x: dict(zip(x, range(len(x))))
        self.w2i = reverse(self.i2w)
        if len(self.w2i) != len(self.i2w):
            print("serious bug: words dumplicated, please check!")

        print("Vocab info: #words %d" % (self.size))


    def word2id(self, xs):
        if isinstance(xs, list):
            return [self.w2i.get(x, self.UNK) for x in xs]
        return self.w2i.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self.i2w[x] for x in xs]
        return self.i2w[xs]

    def save2file(self, outfile):
        with open(outfile, 'w', encoding='utf8') as file:
            for id, word in enumerate(self.i2w):
                if id > self.UNK: file.write(word + '\n')
            file.close()

    @property
    def size(self):
        return len(self.i2w)

import argparse
import pickle

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--infile', default='ch-en-model/tgt_vocab')
    argparser.add_argument('--outfile', default='ch-en-model/tgt_vocab.txt')

    args, extra_args = argparser.parse_known_args()

    vocab = pickle.load(open(args.infile, 'rb'))
    vocab.save2file(args.outfile)
