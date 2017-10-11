import os
import dynet
import seq2seq
import rnnlm as rnnlm
import util

BEGIN_TOKEN = '<s>'
END_TOKEN = '<e>'

def load_vocab(args):
    reader = util.get_reader(args.reader_mode)(args.train, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN)
    vocab = util.Vocab.load_from_corpus(reader, remake=args.rebuild_vocab)
    vocab.START_TOK = vocab[BEGIN_TOKEN]
    vocab.END_TOK = vocab[END_TOKEN]
    vocab.add_unk(args.unk_thresh)

    return vocab


def load_model(path, args=None):
    if not args:
        if not os.path.exists(path): raise Exception("Model "+path+" does not exist")
        with open(path+"/args", "r") as f: args = pickle.load(f)

    model = dynet.Model()
    RNNModel = rnnlm.get_lm(args.model)

    vocab = load_vocab(args)

    if args.s2s:
        print "loading s2s..."
        s2s = seq2seq.get_s2s(args.s2s_type).load(model, args.s2s)

        pron_dict = util.PronDict(model, s2s)
        #print "getting prons for train data"
        #pron_dict.add_prons(train_data)
        #print "getting prons for valid data"
        #pron_dict.add_prons(valid_data)
        
        lm = RNNModel(model, vocab, pron_dict, s2s, args)
    else:
        lm = RNNModel(model, vocab, args)

    if not args:
        lm.m.load(path + "/params")
    else:
        lm.m.load(path)

    return lm, vocab

def gen_samples(lm, vocab, n, path="test_samples.txt"):

    samples = []

    with open(path, "w") as out:
        for i in range(n):
            sample = lm.sample(first=BEGIN_TOKEN,stop=END_TOKEN,nchars=1000)
            samples.append(sample)
            out.write(vocab.pp(sample, ' ') + "\n")

    return samples
