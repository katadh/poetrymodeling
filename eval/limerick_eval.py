# requires pip install pronouncing
import pronouncing as pro
import re
import os
import random

random.seed(78789)

desired_syllables = [9.0, 9.0, 6.0, 6.0, 9.0]
rhyming_line_sets = [[0, 1, 4], [2, 3]]
punc = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'

no_phones_file = "no_phone_words.txt"
no_rhymes_file = "no_rhyme_words.txt"

def log_out_word(word, word_type):

    global no_phones_file
    global no_rhymes_file

    if word_type == "rhyme":
        out_words_file = no_phones_file
    else:
        out_words_file = no_rhymes_file

    with open(out_words_file, 'a+') as out_words:
        current_words = out_words.readlines()
        word += '\n'
        if word not in current_words:
            out_words.write(word)

def line_syllables(line):

    global punc
    words = line.split()
    #phones = [pro.phones_for_word(word)[0] for word in words]
    line_phones = []
    num_syllables = 0
    for word in words:
        word = word.translate(None, punc)
        word_phones = pro.phones_for_word(word)
        if word_phones == []:
            # add syllables with the average of 3 letters per syllable
            num_syllables += len(word) / 3
            log_out_word(word, "phone")
        else:
            line_phones.append(word_phones[0]) 
    
    #print [pro.syllable_count(phone_seq) for phone_seq in line_phones]
    num_syllables += sum([pro.syllable_count(phone_seq) for phone_seq in line_phones])
    return num_syllables

def line_syllable_error(lines):

    global desired_syllables

    errors = []
    for i in range(len(lines)):
        obs_syllables = line_syllables(lines[i])
        if i < len(desired_syllables):
            errors.append(abs(obs_syllables - desired_syllables[i]))
        else:
            errors.append(obs_syllables)

    if len(lines) < len(desired_syllables):
        "Too few lines"
        for i in range(len(lines),len(desired_syllables)):
            errors.append(desired_syllables[i])

    return errors


def total_syllable_error(limerick):

    lines = limerick.split('\n')
    #lines = re.split('<br.>', limerick)

    line_errors = line_syllable_error(lines)

    return sum(line_errors)

def average_line_syllable_error(limericks):

    print "getting average line syllable error"

    total_line_errors = []
    line_counts = []
    for limerick in limericks:
        lim_lines = limerick.split('\n')
        #lim_lines = re.split('<br.>', limerick)

        line_errors = line_syllable_error(lim_lines)

        # this does not assume that every poem has the same number of lines
        for i in range(len(line_errors)):
            if i >= len(total_line_errors):
                total_line_errors.append(line_errors[i])
                line_counts.append(1.0)
            else:
                total_line_errors[i] += line_errors[i]
                line_counts[i] += 1.0

    return [error / count for error, count in zip(total_line_errors, line_counts)]


def average_total_syllable_error(limericks):

    print "getting average total syllable error"

    total_error = 0.0
    for limerick in limericks:
        total_error += total_syllable_error(limerick)

    return 1.0 * total_error / len(limericks)

def rhyming_error(limerick):

    global rhyming_line_sets
    global punc
    
    total_rhyme_error = 0

    lines = limerick.split('\n')
    if len(lines) > 5:
        total_rhyme_error += len(lines) - 5
    #lines = re.split('<br.>', limerick)

    line_words = [line.split() for line in lines]

    for rhyme_lines in rhyming_line_sets:

        min_error = float("inf")
        for i in rhyme_lines:
            #print "i: ", i
            if i < len(line_words) and line_words[i] != []:
                word = line_words[i][-1].translate(None, punc)
                rhymes = pro.rhymes(word)
            else:
                rhymes = []
                #print "given: ", word
            if rhymes == []:
                log_out_word(word, "rhyme")
            #print rhymes

            rhyme_error = 0
            index = rhyme_lines.index(i)
            for j in (rhyme_lines[:index] + rhyme_lines[index+1:]):
                #print "j: ", j
                if j < len(line_words) and line_words[j] != []:
                    query_word = line_words[j][-1].translate(None, punc)
                else:
                    query_word = ""
                #print "compare: ", query_word
                #print query_word
                if query_word not in rhymes:
                    rhyme_error += 1
                    #print rhyme_error

            if rhyme_error < min_error:
                min_error = rhyme_error

        total_rhyme_error += min_error

    return total_rhyme_error

def average_rhyming_error(limericks):

    print "getting average rhyming error"

    total_error = 0.0
    for limerick in limericks:
        #print limerick
        total_error += rhyming_error(limerick)

    return 1.0 * total_error / len(limericks)
    

def eval_limerick_files(dir_path, num_lim):

    lim_files = os.listdir(dir_path)

    limericks = []
    for i in range(num_lim):
        lim_file = random.choice(lim_files)

        with open (dir_path + lim_file) as lim:
            text = lim.read().decode('utf8').encode('ascii', 'ignore').lower()
            limericks.append(text)

        lim_files.remove(lim_file)

    #print limericks

    avg_line_struct_error = average_line_syllable_error(limericks)
    avg_total_struct_error = average_total_syllable_error(limericks)
    avg_rhyme_error = average_rhyming_error(limericks)

    return avg_line_struct_error, avg_total_struct_error, avg_rhyme_error

def eval_file_limericks(path):

    limericks = []

    with open(path, 'r') as lims_file:
        limericks = lims_file.readlines()

    limericks = [lim.replace(' <br> ', '\n').encode('ascii', 'ignore').lower() for lim in limericks]

    avg_line_struct_error = average_line_syllable_error(limericks)
    avg_total_struct_error = average_total_syllable_error(limericks)
    avg_rhyme_error = average_rhyming_error(limericks)

    return avg_line_struct_error, avg_total_struct_error, avg_rhyme_error
