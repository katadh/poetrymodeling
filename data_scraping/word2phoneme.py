import dynet as pc
import random
import util
import subprocess

EOS = "<EOS>"
# characters = list("abcdefghijklmnopqrstuvwxyz ")
# characters.append(EOS)
# 
# int2char = list(characters)
# char2int = {c:i for i,c in enumerate(characters)}
TRAIN_FILEPATH_SRC = "/Users/chaitanya/Desktop/NMT/nmt-tips/data/train.en"
DEV_FILEPATH_SRC = "/Users/chaitanya/Desktop/NMT/nmt-tips/data/dev.en"
TEST_FILEPATH_SRC = "/Users/chaitanya/Desktop/NMT/nmt-tips/data/test.en"
TRAIN_FILEPATH_TARGET = "/Users/chaitanya/Desktop/NMT/nmt-tips/data/train.ja"
DEV_FILEPATH_TARGET = "/Users/chaitanya/Desktop/NMT/nmt-tips/data/dev.ja"
TEST_FILEPATH_TARGET = "/Users/chaitanya/Desktop/NMT/nmt-tips/data/test.ja"

c = util.CorpusReader(TRAIN_FILEPATH_SRC)
v = util.Vocab.fromCorpus(c)
VOCAB_SIZE = v.size()
LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 300
STATE_SIZE = 128
ATTENTION_SIZE = 64

# DEV SET
# LEARNING RATE
# MINIBATCH SIZE
# EPOCHS
# EVAL-EVERY
# RATE DECAY
# GRADIENT CLIPPING
# SUBWORD NMT
# DROPOUT
# FUNCTION TO FIND VOCAB SIZE
# COVERAGE EMBEDDING
# WORD ALIGNMENTS

model = pc.Model()


#Bidirectional Encoder LSTM
print "Adding Forward encoder LSTM parameters"
enc_fwd_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
print "Adding Backward encoder LSTM parameters"
enc_bwd_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

#Decoder LSTM
print "Adding decoder LSTM parameters"
dec_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2, STATE_SIZE, model)

print "Adding lookup parameters"
#Lookup parameters
lookup = model.add_lookup_parameters( (VOCAB_SIZE, EMBEDDINGS_SIZE))

#Attention parameters
print "Adding Attention Parameters"
attention_w1 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*2))
attention_w2 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
attention_v = model.add_parameters( (1, ATTENTION_SIZE))


#Decoder weight and bias
print "Adding Decoder weight"
decoder_w = model.add_parameters( (VOCAB_SIZE, STATE_SIZE))
print "Adding Decoder bias" 
decoder_b = model.add_parameters( (VOCAB_SIZE))


def embed_sentence(sentence):
    sentence = [EOS] + list(sentence) + [EOS]
    sentence = [lookup[word] for word in sentence]

    global lookup

    return [lookup[word] for word in sentence]


def run_lstm(init_state, input_vecs):
    s = init_state

    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode_sentence(enc_fwd_lstm, enc_bwd_lstm, sentence):
    sentence_rev = list(reversed(sentence))

    fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), sentence)
    bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [pc.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
    print "Vectors: ", len(fwd_vectors), len(bwd_vectors)
    return vectors


def attend(input_vectors, state):
    global attention_w1
    global attention_w2
    global attention_v
    w1 = pc.parameter(attention_w1)
    w2 = pc.parameter(attention_w2)
    v = pc.parameter(attention_v)
    attention_weights = []
    
    #
#     w2dt = w2*pc.concatenate(list(state.s()))
    w2dt = w2*pc.concatenate(list(state.s()))
    for input_vector in input_vectors:
        attention_weight = v*pc.tanh(w1*input_vector + w2dt)
        attention_weights.append(attention_weight)
    attention_weights = pc.softmax(pc.concatenate(attention_weights))
    output_vectors = pc.esum([vector*attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])
    return output_vectors


def decode(dec_lstm, vectors, output):
    output = [EOS] + list(output) + [EOS]
    output = [char2int[c] for c in output]

    w = pc.parameter(decoder_w)
    b = pc.parameter(decoder_b)

    s = dec_lstm.initial_state().add_input(pc.vecInput(STATE_SIZE*2))

    loss = []
    for word in output:
        vector = attend(vectors, s)

        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = pc.softmax(out_vector)
        loss.append(-pc.log(pc.pick(probs, char)))
    loss = pc.esum(loss)
    return loss

def generate(input, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    def sample(probs):
        rnd = random.random()
        for i, p in enumerate(probs):
            rnd -= p
            if rnd <= 0: break
        return i

    embedded = embed_sentence(input)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = pc.parameter(decoder_w)
    b = pc.parameter(decoder_b)

    s = dec_lstm.initial_state().add_input(pc.vecInput(STATE_SIZE * 2))
    out = ''
    count_EOS = 0
    for i in range(len(input)*2):
        if count_EOS == 2: break
        vector = attend(encoded, s)

        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = pc.softmax(out_vector)
        probs = probs.vec_value()
        next_word = sample(probs)
        if next_word == EOS:
            count_EOS += 1
            continue

        out += next_word
    return out


def get_loss(input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    #Renew computation graph 
    pc.renew_cg()
    embedded = embed_sentence(input_sentence)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)
    return decode(dec_lstm, encoded, output_sentence)

# def dropout():
    


def train(model, source, target):
    trainer = pc.SimpleSGDTrainer(model)
    for i in xrange(600):
        loss = get_loss(source, target, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        if i % 20 == 0:
            print "Loss:" , loss_value
            print generate(source, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)


train(model, "and now you are on your own", "y ahora se encuentra en su propia")

