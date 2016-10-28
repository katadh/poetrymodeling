from nltk.tokenize import TreebankWordTokenizer

f_r = open("path/to/file/for/tokenization")
lines = [line.lower() for line in f_r]
f_r.close()

f_w = open('path/to/tokenized/file', 'w')
count = 0
for sent in lines:
    s = TreebankWordTokenizer().tokenize(sent)
    f_w.write(" ".join(s))
    if (count!=len(lines)-1):
        f_w.write("\n")
    count+=1
f_w.close()