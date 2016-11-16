import dynet
import seq2seq
import util
import rnnlm as rnnlm
import argparse, random, time, sys, math
from itertools import combinations
sys.setrecursionlimit(5000) # sometimes we need to recurse a lot
random.seed(78789)

parser = argparse.ArgumentParser()

## need to have this dummy argument for dynet
parser.add_argument("--dynet-mem")

## locations of data
parser.add_argument("--train", default="../data/ohhla")
parser.add_argument("--valid")
parser.add_argument("--test")

## alternatively, load one dataset and split it
parser.add_argument("--split_train", action='store_true')
parser.add_argument("--percent_valid", default=.02, type=float)
parser.add_argument("--percent_test", default=.05, type=float)

## vocab parameters
parser.add_argument('--reader_mode', default="ohhla")
parser.add_argument('--rebuild_vocab', action='store_true')
parser.add_argument('--unk_thresh', default=5, type=int)

## rnn parameters
parser.add_argument("--layers", default=1, type=int)
parser.add_argument("--input_dim", default=10, type=int)
parser.add_argument("--hidden_dim", default=50, type=int)
parser.add_argument("--rnn", default="lstm")

## experiment parameters
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--learning_rate", default=1.0, type=float)
#parser.add_argument("--max_corpus_size", default=100000, type=int)
parser.add_argument("--log_train_every_n", default=100, type=int)
parser.add_argument("--log_valid_every_n", default=5000, type=int)
parser.add_argument("--output")
parser.add_argument('--ignore_parens_perplexity', action='store_true')

## choose what model to use
parser.add_argument("--model", default="baseline")
parser.add_argument("--s2s")

## model-specific parameters
parser.add_argument("--sample_count", default=10, type=int)

args = parser.parse_args()
print "ARGS:", args

if args.rnn == "lstm": args.rnn = dynet.LSTMBuilder
elif args.rnn == "gru": args.rnn = dynet.GRUBuilder
else: args.rnn = dynet.SimpleRNNBuilder

BEGIN_TOKEN = '<s>'
END_TOKEN = '<e>'
reader = util.OHHLACorpusReader(args.train, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN)
vocab = util.Vocab.load_from_corpus(reader, remake=args.rebuild_vocab)
vocab.START_TOK = vocab[BEGIN_TOKEN]
vocab.END_TOK = vocab[END_TOKEN]
vocab.add_unk(args.unk_thresh)

model = dynet.Model()
sgd = dynet.SimpleSGDTrainer(model)

if args.s2s:
    print "loading..."
    args.s2s = seq2seq.Seq2SeqBasic.load(model, args.s2s)

RNNModel = rnnlm.get_lm(args.model)
lm = RNNModel(model, vocab, args)

train_data = list(util.OHHLACorpusReader(args.train, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
if not args.split_train:
    valid_data = list(util.OHHLACorpusReader(args.valid, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
    test_data  = list(util.OHHLACorpusReader(args.test, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
else:
    valid_data = train_data[-int(len(train_data)*(args.percent_valid+args.percent_test)):-int(len(train_data)*args.percent_test)]
    test_data = train_data[-int(len(train_data)*args.percent_test):]
    train_data = train_data[:-int(len(train_data)*(args.percent_valid+args.percent_test))]

if args.output:
    outfile = open(args.output, 'w')
    outfile.write("")
    outfile.close()

special_chars = {tok.s for tok in vocab.tokens if tok.s in {BEGIN_TOKEN,}}
perplexity_denom = lambda x:len([i for i in x if i not in special_chars])

char_count = sent_count = cum_loss = cum_perplexity = 0.0
_start = time.time()
for ITER in range(args.epochs):
    lm.epoch = ITER
    random.shuffle(train_data)

    for i,sent in enumerate(train_data):
        sample_num = 1+i+(len(train_data)*ITER)

        if sample_num % args.log_train_every_n == 0:
            print ITER, sample_num, " ",
            sgd.status()
            print "L:", cum_loss / char_count,
            print "P:", cum_perplexity / sent_count,
            print "T:", (time.time() - _start),
            _start = time.time()
            sample = lm.sample(first=BEGIN_TOKEN,stop=END_TOKEN,nchars=1000)
            if sample: print vocab.pp(sample, ' '),
            char_count = sent_count = cum_loss = cum_perplexity = 0.0
            print
        # end of test logging

        if sample_num % args.log_valid_every_n == 0:
            v_char_count = v_sent_count = v_cum_loss = v_cum_perplexity = 0.0
            v_start = time.time()
            for v_sent in valid_data:
                v_isent = [vocab[w].i for w in v_sent]
                v_loss = lm.BuildLMGraph(v_isent, sent_args={"test":True,
                                                             "special_chars":special_chars})
                v_cum_loss += v_loss.scalar_value()
                v_cum_perplexity += math.exp(v_loss.scalar_value()/perplexity_denom(v_sent))
                v_char_count += perplexity_denom(v_sent)
                v_sent_count += 1
            print "[Validation "+str(sample_num) + "]\t" + \
                  "Loss: "+str(v_cum_loss / v_char_count) + "\t" + \
                  "Perplexity: "+str(v_cum_perplexity / v_sent_count) + "\t" + \
                  "Time: "+str(time.time() - v_start),
            if args.output:
                print "(logging to", args.output + ")"
                with open(args.output, "a") as outfile:
                    outfile.write(str(ITER) + "\t" + \
                                  str(sample_num) + "\t" + \
                                  str(v_cum_loss / v_char_count) + "\t" + \
                                  str(v_cum_perplexity / v_sent_count) + "\n")
            print "\n"            
	 # end of validation logging

        isent = [vocab[w].i for w in sent]
        loss = lm.BuildLMGraph(isent, sent_args={"special_chars":special_chars})
        cum_loss += loss.value()
        cum_perplexity += math.exp(loss.value()/perplexity_denom(sent))
        char_count += perplexity_denom(sent)
        sent_count += 1

        loss.backward()
        sgd.update(args.learning_rate)
        # end of one-sentence train loop
    sgd.update_epoch(args.learning_rate)
    # end of iteration
# end of training loop

t_char_count = t_sent_count = t_cum_loss = t_cum_perplexity = 0.0
t_start = time.time()
for t_sent in test_data:
    t_isent = [vocab[w].i for w in t_sent]
    t_loss = lm.BuildLMGraph(t_isent, sent_args={"test":True, 
                                                 "special_chars":special_chars})
    t_cum_loss += t_loss.scalar_value()
    t_cum_perplexity += math.exp(t_loss.scalar_value()/perplexity_denom(t_sent))
    t_char_count += perplexity_denom(t_sent)
    t_sent_count += 1
print "[Test]\t" + \
      "Loss: "+str(t_cum_loss / t_char_count) + "\t" + \
      "Perplexity: "+str(t_cum_perplexity / t_sent_count) + "\t" + \
      "Time: "+str(time.time() - t_start),
if args.output:
    print "(logging to", args.output + ")"
    with open(args.output, "a") as outfile:
        outfile.write(str("TEST") + "\t" + \
                      str("TEST") + "\t" + \
                      str(t_cum_loss / t_char_count) + "\t" + \
                      str(t_cum_perplexity / t_sent_count) + "\n")
    
  

