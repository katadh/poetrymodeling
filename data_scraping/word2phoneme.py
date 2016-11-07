from itertools import izip
import dynet as pc
import random
import util
import subprocess
import numpy as np
import string
import split_cmu_dict



EOS = "<s>"

TRAIN_FILEPATH_SRC = "cmu_dict_train"
SYMBOLS_SRC = "/Users/chaitanya/Desktop/Study/10701/Project/cmudict-0.7b.symbols"
DEV_FILEPATH_SRC = "cmu_dict_dev"
TEST_FILEPATH_SRC = "cmu_dict_test"

EPOCHS = 10

## SOURCE
characters = list(string.ascii_uppercase) + [" "]
characters += list(string.punctuation)
characters.append(EOS)
int2char = list(characters)
char2int = {c:i for i,c in enumerate(characters)}

VOCAB_SIZE_SRC = len(characters)
 

## TARGET (SYMBOLS) 
d = util.CorpusReader(SYMBOLS_SRC)
vt = util.Vocab.from_corpus(d)
VOCAB_SIZE_TGT = vt.size()
#ADD EOS
vt.w2i[EOS] = VOCAB_SIZE_TGT
vt.i2w[VOCAB_SIZE_TGT] = EOS
VOCAB_SIZE_TGT = VOCAB_SIZE_TGT + 1

LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 300
STATE_SIZE = 128
ATTENTION_SIZE = 64
MINIBATCH_SIZE = 256

# DEV SET
# LEARNING RATE
# MINIBATCH SIZE - batch across instances
#Group sentences into a mini batch (optionally, for
#efficiency group sentences by length)
#Select the "t"th word in each sentence, and send
#them to the lookup and loss functions

# EPOCHS
# EVAL-EVERY
# RATE DECAY
# GRADIENT CLIPPING
# SUBWORD NMT
# DROPOUT
# COVERAGE EMBEDDING
# WORD ALIGNMENTS -  FASTALIGN
# STACKED RNN


def read(fname):
	"""`
	Read a file where each line is of the form "word1 word2 ..."
	Yields lists of the form [word1, word2, ...]
	"""
	cmu_dict = split_cmu_dict.load_dict(TRAIN_FILEPATH_SRC)
	for mapping in cmu_dict:
		yield word, phonemes

model = pc.Model()

# Bidirectional Encoder LSTM
print "Adding Forward encoder LSTM parameters"
enc_fwd_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
print "Adding Backward encoder LSTM parameters"
enc_bwd_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

#Decoder LSTM
print "Adding decoder LSTM parameters"
dec_lstm = pc.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2, STATE_SIZE, model)

print "Adding lookup parameters"
#Lookup parameters
lookup_src = model.add_lookup_parameters( (VOCAB_SIZE_SRC, EMBEDDINGS_SIZE))
lookup_tgt = model.add_lookup_parameters( (VOCAB_SIZE_TGT, EMBEDDINGS_SIZE))

#Attention parameters
print "Adding Attention Parameters"
attention_w1 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*2))
attention_w2 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
attention_v = model.add_parameters( (1, ATTENTION_SIZE))


#Decoder weight and bias
print "Adding Decoder weight"
decoder_w = model.add_parameters( (VOCAB_SIZE_TGT, STATE_SIZE))
print "Adding Decoder bias" 
decoder_b = model.add_parameters( (VOCAB_SIZE_TGT))



def embed_word(word):

	global lookup_src

	word = [EOS] + list(word) + [EOS]
	word = [lookup_src[char2int[char]] for char in word if char!=" " and char not in string.digits ]

	return word


def run_lstm(init_state, input_vecs):
	s = init_state

	out_vectors = []
	for vector in input_vecs:
		s = s.add_input(vector)
		out_vector = s.output()
		out_vectors.append(out_vector)
	return out_vectors


def encode_word(enc_fwd_lstm, enc_bwd_lstm, word):
	
	word_rev = [lookup_src[char2int[EOS]]] + list(reversed(word))  + [lookup_src[char2int[EOS]]]

	fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), word)
	bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), word_rev)
	bwd_vectors = list(reversed(bwd_vectors))
	vectors = [pc.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
	# print "Vectors: ", len(fwd_vectors), len(bwd_vectors)
	return vectors


def attend(input_vectors, state):
	global attention_w1
	global attention_w2
	global attention_v
	w1 = pc.parameter(attention_w1)
	w2 = pc.parameter(attention_w2)
	v = pc.parameter(attention_v)
	attention_weights = []
	w2dt = w2*pc.concatenate(list(state.s()))

	for input_vector in input_vectors:
		attention_weight = v*pc.tanh(w1*input_vector + w2dt)
		attention_weights.append(attention_weight)
	attention_weights = pc.softmax(pc.concatenate(attention_weights))
	output_vectors = pc.esum([vector*attention_weight for vector, attention_weight in zip(input_vectors, attention_weights)])
	return output_vectors


def decode(dec_lstm, vectors, output):

	output = [EOS] + list(output) + [EOS]
	idx = [vt.w2i[symbol] for symbol in output]
	output = [lookup_tgt[vt.w2i[symbol]] for symbol in output]
	
	w = pc.parameter(decoder_w)
	b = pc.parameter(decoder_b)

	s = dec_lstm.initial_state().add_input(pc.vecInput(STATE_SIZE*2))

	loss = []
	for symbol in idx:
		vector = attend(vectors, s)
		s = s.add_input(vector)
		out_vector = w * s.output() + b
		probs = pc.softmax(out_vector)
		loss.append(-pc.log(pc.pick(probs, symbol)))
	loss = pc.esum(loss)
	return loss


def generate(word, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
	def sample(probs):
		rnd = random.random()
		for i, p in enumerate(probs):
			rnd -= p
			if rnd <= 0: break
		return i

	embedded = embed_word(word)
	encoded = encode_word(enc_fwd_lstm, enc_bwd_lstm, embedded)

	w = pc.parameter(decoder_w)
	b = pc.parameter(decoder_b)

	s = dec_lstm.initial_state().add_input(pc.vecInput(STATE_SIZE * 2))
	out = ''
	count_EOS = 0
	for i in range(len(word)*2):
		if count_EOS == 2: break
		vector = attend(encoded, s)

		s = s.add_input(vector)
		out_vector = w * s.output() + b
		probs = pc.softmax(out_vector)
		probs = probs.vec_value()
		next_symbol = sample(probs)
		if vt.i2w[next_symbol] == EOS:
			count_EOS += 1
			continue

		out += vt.i2w[next_symbol]
	return out
	

def get_loss(input, output, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
	#Renew computation graph 
	pc.renew_cg()
	embedded = embed_word(input)
	encoded = encode_word(enc_fwd_lstm, enc_bwd_lstm, embedded)
	return decode(dec_lstm, encoded, output)

# def dropout():

def train(model, trainer, source, target):
	loss = get_loss(source, target, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
	loss_value = loss.value()
	loss.backward()
	trainer.update()
	return loss_value


def trainExample(model):
	#SGD Trainer    
	trainer = pc.4(model)
	symbols = []
	# words, symbols = read(TRAIN_FILEPATH_SRC)

	for i in xrange(EPOCHS):
		for src, target in read(TRAIN_FILEPATH_SRC):
				loss_value = train(model, trainer, src, target)
		if (i%100==0):
			print "Epoch: ", i , "Loss: " , loss_value

		model.save(MODEL_PATH)
		evalDevSet(model)

def evalDevSet(model):


def loadModel(modelPath):
	lookup_src = model.add_lookup_parameters( (VOCAB_SIZE_SRC, EMBEDDINGS_SIZE))
	lookup_tgt = model.add_lookup_parameters( (VOCAB_SIZE_TGT, EMBEDDINGS_SIZE))
	model.load(modelPath)

trainExample(model)
symbols = generate("abba".upper(), enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
print symbols
loadModel(MODEL_PATH)
