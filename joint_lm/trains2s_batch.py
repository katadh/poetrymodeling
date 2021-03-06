import dynet
import argparse
import util
import time
import random
import sys
import seq2seq

# input args

parser = argparse.ArgumentParser()

## need to have this dummy argument for dynet
parser.add_argument("--dynet-mem")
parser.add_argument("--dynet-gpu")

## locations of data
parser.add_argument("--train")
parser.add_argument("--valid")

## alternatively, load one dataset and split it
parser.add_argument("--percent_valid", default=1000, type=float)

## vocab parameters
parser.add_argument('--rebuild_vocab', action='store_true')
parser.add_argument('--unk_thresh', default=1, type=int)

## rnn parameters
parser.add_argument("--layers", default=1, type=int)
parser.add_argument("--input_dim", default=10, type=int)
parser.add_argument("--hidden_dim", default=50, type=int)
parser.add_argument("--attention_dim", default=50, type=int)
parser.add_argument("--rnn", default="lstm")

## experiment parameters
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--learning_rate", default=1.0, type=float)
parser.add_argument("--log_train_every_n", default=100, type=int)
parser.add_argument("--log_valid_every_n", default=5000, type=int)
parser.add_argument("--output")

## choose what model to use
parser.add_argument("--model", default="basic")
parser.add_argument("--reader_mode")
parser.add_argument("--load")
parser.add_argument("--save")

## model-specific parameters
parser.add_argument("--beam_size", default=3, type=int)
parser.add_argument("--minibatch_size", default=100, type=int)

args = parser.parse_args()
print "ARGS:", args

if args.rnn == "lstm": args.rnn = dynet.LSTMBuilder
elif args.rnn == "gru": args.rnn = dynet.GRUBuilder
else: args.rnn = dynet.SimpleRNNBuilder

BEGIN_TOKEN = '<s>'
END_TOKEN = '<e>'

# define model

model = dynet.Model()
sgd = dynet.SimpleSGDTrainer(model)

S2SModel = seq2seq.get_s2s(args.model)
if args.load:
    print "loading..."
    s2s = S2SModel.load(model, args.load)
    src_vocab = s2s.src_vocab
    tgt_vocab = s2s.tgt_vocab
else:
    print "fresh model. getting vocab...",
    src_reader = util.get_reader(args.reader_mode)(args.train, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN)
    src_vocab = util.Vocab.load_from_corpus(src_reader, remake=args.rebuild_vocab, src_or_tgt="src")
    src_vocab.START_TOK = src_vocab[BEGIN_TOKEN]
    src_vocab.END_TOK = src_vocab[END_TOKEN]
    src_vocab.add_unk(args.unk_thresh)
    tgt_reader = util.get_reader(args.reader_mode)(args.train, mode=args.reader_mode, end=END_TOKEN)
    tgt_vocab = util.Vocab.load_from_corpus(tgt_reader, remake=args.rebuild_vocab, src_or_tgt="tgt")
    tgt_vocab.END_TOK = tgt_vocab[END_TOKEN]
    tgt_vocab.add_unk(args.unk_thresh)
    print "making model..."
    s2s = S2SModel(model, src_vocab, tgt_vocab, args)
    print "...done."

# load corpus

print "loading corpus..."
train_data = list(util.get_reader(args.reader_mode)(args.train, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
if args.valid:
    valid_data = list(util.get_reader(args.reader_mode)(args.valid, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
else:
    if args.percent_valid > 1: cutoff = args.percent_valid
    else: cutoff = int(len(train_data)*(args.percent_valid))
    valid_data = train_data[-cutoff:]
    train_data = train_data[:-cutoff]
    print "Train set of size", len(train_data), "/ Validation set of size", len(valid_data)
print "done."

if args.output:
    outfile = open(args.output, 'w')
    outfile.write("")
    outfile.close()



# train_data.sort(key=lambda x: -len(x))
# valid_data.sort(key=lambda x: -len(x))

# run training loop

char_count = sent_count = cum_loss = 0.0


# Store starting index of each minibatch
train_order = [x*args.minibatch_size for x in range(len(train_data)/args.minibatch_size + 1)]
valid_order = [x*args.minibatch_size for x in range(len(valid_data)/args.minibatch_size + 1)]


_start = time.time()
try:
    for ITER in range(args.epochs):
        s2s.epoch = ITER
        random.shuffle(train_order)

        for i, sid in enumerate(train_order):

            src = train_data[sid]
            tgt = train_data[sid]

            src = [src_vocab[s] for s in src]
            tgt = [tgt_vocab[s] for s in tgt]
            batched_src = src[sid : sid + args.minibatch_size]
            batched_tgt = tgt[sid : sid + args.minibatch_size]

            sample_num = 1 + sid + (len(train_data)*ITER)

            # print sample_num, src_vocab.pp(src, ' '), tgt_vocab.pp(tgt, ' ')

            if sample_num % args.log_train_every_n == 0:
                print ITER, sample_num, " ",
                sgd.status()
                print "L:", cum_loss / char_count if char_count != 0 else None,
                print "T:", (time.time() - _start),
                _start = time.time()
                # sample = lm.beam_search_generate(src, beam_n=args.beam_size)
                sample = s2s.generate(src, sampled=False)
                if sample: print src_vocab.pp(src, ' '), tgt_vocab.pp(tgt, ' '), tgt_vocab.pp(sample, ' '),
                char_count = sent_count = cum_loss = cum_bleu = 0.0
                print
            # end of test logging

            if sample_num % args.log_valid_every_n == 0:
                v_char_count = v_sent_count = v_cum_loss = v_cum_bleu = v_cum_em = v_cum_perp = 0.0
                v_start = time.time()
                for vid in valid_order:
                    src = valid_data[vid]
                    tgt = valid_data[vid]
                    batched_v_src = src[vid : vid + args.minibatch_size]
                    batched_v_tgt = tgt[vid : vid + args.minibatch_size]
                    # v_src = [src_vocab[tok] for tok in v_src]
                    # v_tgt = [tgt_vocab[tok] for tok in v_tgt]
                    v_loss = s2s.get_batch_loss(batched_v_src, batched_v_tgt)
                    v_cum_loss += v_loss.scalar_value()
                    v_cum_perp += s2s.get_perplexity(v_src, v_tgt)
                    v_cum_em += s2s.get_em(v_src, v_tgt)
                    # v_cum_bleu += s2s.get_bleu(v_src, v_tgt, args.beam_size)
                    v_char_count += len(v_tgt)-1
                    v_sent_count += 1
                print "[Validation "+str(sample_num) + "]\t" + \
                      "Loss: "+str(v_cum_loss / v_char_count) + "\t" + \
                      "Perp: "+str(v_cum_perp / v_sent_count) + "\t" + \
                      "BLEU: "+str(v_cum_bleu / v_sent_count) + "\t" + \
                      "EM: "  +str(v_cum_em   / v_sent_count) + "\t" + \
                      "Time: "+str(time.time() - v_start),
                if args.output:
                    print "(logging to", args.output + ")"
                    with open(args.output, "a") as outfile:
                        outfile.write(str(ITER) + "\t" + \
                                      str(sample_num) + "\t" + \
                                      str(v_cum_loss / v_char_count) + "\t" + \
                                      str(v_cum_perp / v_sent_count) + "\t" + \
                                      str(v_cum_em   / v_sent_count) + "\t" + \
                                      str(v_cum_bleu / v_sent_count) + "\n")
                print "\n"
                if args.save:
                    print "saving checkpoint..."
                    s2s.save(args.save+".checkpoint")
            # end of validation logging
            
            # loss = s2s.get_loss(src, tgt)
            loss = s2s.get_batch_loss(batched_src, batched_tgt)
            cum_loss += loss.value()
            char_count += [len(tgt_sent)-1 for tgt_sent in tgt[sid : sid + args.minibatch_size]]
            sent_count += args.minibatch_size

            loss.backward()
            sgd.update(args.learning_rate)

            ### end of one-sentence train loop
        sgd.update_epoch(args.learning_rate)
        ### end of iteration
    ### end of training loop
except KeyboardInterrupt:
    if args.save:
        print "saving..."
        s2s.save(args.save)
        sys.exit()

if args.save:
    print "saving..."
    s2s.save(args.save)
