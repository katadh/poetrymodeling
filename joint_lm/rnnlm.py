import dynet
import random
import util
import math
import seq2seq
###########################################################################
class RNNLanguageModel(object):
    name = "template"

    def __init__(self, model, vocab, args):
        self.model = model
        self.vocab = vocab
        self.args = args

    def BuildLMGraph(self, sent, sent_args=None):
        pass

    def sample(self, first=0, stop=-1, nchars=100):
        pass

def get_lm(name):
    for c in util.itersubclasses(RNNLanguageModel):
        if c.name == name: return c
    raise Exception("no language model found with name: " + name)
##########################################################################

class BaselineRNNLM(RNNLanguageModel):
    name = "baseline"

    def __init__(self, model, vocab, args):
        self.m = model
        self.vocab = vocab
        self.args = args

        self.rnn = args.rnn(args.layers, args.input_dim, args.hidden_dim, model)

        self.lookup = model.add_lookup_parameters((vocab.size, args.input_dim))
        self.R = model.add_parameters((vocab.size, args.hidden_dim))
        self.bias = model.add_parameters((vocab.size,))

    def BuildLMGraph(self, sent, sent_args=None):
        dynet.renew_cg()
        init_state = self.rnn.initial_state()

        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        errs = [] # will hold expressions
        state = init_state

        for (cw,nw) in zip(sent,sent[1:]):
            x_t = self.lookup[cw]
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = dynet.pickneglogsoftmax(r_t, int(nw))
            errs.append(err)
        nerr = dynet.esum(errs)
        return nerr

    def sample(self, first=0, stop=-1, nchars=100):
        first = self.vocab[first].i
        stop = self.vocab[stop].i

        res = [first]
        dynet.renew_cg()
        state = self.rnn.initial_state()

        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        cw = first
        while True:
            x_t = self.lookup[cw]
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            scores = r_t.vec_value()
            if self.vocab.unk is not None:
                ydist = util.softmax(scores[:self.vocab.unk.i]+scores[self.vocab.unk.i+1:]) # remove UNK
                dist = ydist[:self.vocab.unk.i].tolist()+[0]+ydist[self.vocab.unk.i:].tolist()
            else:
                ydist = util.softmax(scores)
                dist = ydist
            rnd = random.random()
            for i,p in enumerate(dist):
                rnd -= p
                if rnd <= 0: break
            res.append(i)
            cw = i
            if cw == stop: break
            if nchars and len(res) > nchars: break
        return res

class BasicJointRNNLM(RNNLanguageModel):
    name = "joint"

    def __init__(self, model, vocab, args):
        self.m = model
        self.vocab = vocab
        self.args = args

        self.s2s = args.s2s
        self.rnn = args.rnn(args.layers, self.s2s.args.hidden_dim*2 + args.input_dim, args.hidden_dim, model)

        self.lookup = model.add_lookup_parameters((vocab.size, args.input_dim))
        self.R = model.add_parameters((vocab.size, args.hidden_dim))
        self.bias = model.add_parameters((vocab.size,))

    def BuildLMGraph(self, sent, sent_args=None):
        dynet.renew_cg()
        init_state = self.rnn.initial_state()

        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        errs = [] # will hold expressions
        state = init_state

        for (cw,nw) in zip(sent,sent[1:]):
            cw = self.vocab[cw]
            nw = self.vocab[nw]

            spelling = [self.s2s.src_vocab[letter] for letter in cw.s.upper()]
            embedded_spelling = self.s2s.embed_seq(spelling)
            pron_vector = self.s2s.encode_seq(embedded_spelling)[-1]
            fpv = dynet.nobackprop(pron_vector)

            x_t = dynet.concatenate([self.lookup[cw.i], fpv])
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = dynet.pickneglogsoftmax(r_t, int(nw.i))
            errs.append(err)
        nerr = dynet.esum(errs)
        return nerr

    def sample(self, first=0, stop=-1, nchars=100):
        first = self.vocab[first].i
        stop = self.vocab[stop].i

        res = [first]
        dynet.renew_cg()
        state = self.rnn.initial_state()

        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        cw = first
        while True:
            spelling = [self.s2s.src_vocab[letter] for letter in self.vocab[cw].s.upper()]
            embedded_spelling = self.s2s.embed_seq(spelling)
            pron_vector = self.s2s.encode_seq(embedded_spelling)[-1]

            x_t = dynet.concatenate([self.lookup[cw], pron_vector])
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            scores = r_t.vec_value()
            if self.vocab.unk is not None:
                ydist = util.softmax(scores[:self.vocab.unk.i]+scores[self.vocab.unk.i+1:]) # remove UNK
                dist = ydist[:self.vocab.unk.i].tolist()+[0]+ydist[self.vocab.unk.i:].tolist()
            else:
                ydist = util.softmax(scores)
                dist = ydist
            rnd = random.random()
            for i,p in enumerate(dist):
                rnd -= p
                if rnd <= 0: break
            res.append(i)
            cw = i
            if cw == stop: break
            if nchars and len(res) > nchars: break
        return res

class PhonemeOnlyRNNLM(RNNLanguageModel):
    name = "prononly"

    def __init__(self, model, vocab, args):
        self.m = model
        self.vocab = vocab
        self.args = args

        self.s2s = args.s2s
        self.rnn = args.rnn(args.layers, self.s2s.args.hidden_dim*2, args.hidden_dim, model)

        self.R = model.add_parameters((vocab.size, args.hidden_dim))
        self.bias = model.add_parameters((vocab.size,))

    def BuildLMGraph(self, sent, sent_args=None):
        dynet.renew_cg()
        init_state = self.rnn.initial_state()

        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        errs = [] # will hold expressions
        state = init_state

        for (cw,nw) in zip(sent,sent[1:]):
            cw = self.vocab[cw]
            nw = self.vocab[nw]

            spelling = [self.s2s.src_vocab[letter] for letter in cw.s.upper()]
            embedded_spelling = self.s2s.embed_seq(spelling)
            pron_vector = self.s2s.encode_seq(embedded_spelling)[-1]
            fpv = dynet.nobackprop(pron_vector)

            x_t = fpv
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = dynet.pickneglogsoftmax(r_t, int(nw.i))
            errs.append(err)
        nerr = dynet.esum(errs)
        return nerr

    def sample(self, first=0, stop=-1, nchars=100):
        first = self.vocab[first].i
        stop = self.vocab[stop].i

        res = [first]
        dynet.renew_cg()
        state = self.rnn.initial_state()

        R = dynet.parameter(self.R)
        bias = dynet.parameter(self.bias)
        cw = first
        while True:
            spelling = [self.s2s.src_vocab[letter] for letter in self.vocab[cw].s.upper()]
            embedded_spelling = self.s2s.embed_seq(spelling)
            pron_vector = self.s2s.encode_seq(embedded_spelling)[-1]

            x_t = pron_vector
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            scores = r_t.vec_value()
            if self.vocab.unk is not None:
                ydist = util.softmax(scores[:self.vocab.unk.i]+scores[self.vocab.unk.i+1:]) # remove UNK
                dist = ydist[:self.vocab.unk.i].tolist()+[0]+ydist[self.vocab.unk.i:].tolist()
            else:
                ydist = util.softmax(scores)
                dist = ydist
            rnd = random.random()
            for i,p in enumerate(dist):
                rnd -= p
                if rnd <= 0: break
            res.append(i)
            cw = i
            if cw == stop: break
            if nchars and len(res) > nchars: break
        return res
