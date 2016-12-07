# requires pip install pronouncing
import pronouncing as pro
import re

desired_syllables = [9.0, 9.0, 6.0, 6.0, 9.0]
rhyming_line_sets = [[1, 2, 5], [3, 4]]

def line_syllables(line):

    phones = [pro.phones_for_word(word)[0] for word in line.split()]
    num_syllables = sum([pro.syllable_count(phone_seq) for phone_seq in phones])
    return num_syllables

def line_syllable_error(lines):

    global desired_syllables

    errors = []
    for i in range(len(lines)):
        obs_syllables = line_syllables(line[i])
        if i < len(desired_syllables):
            errors.append(abs(obs_syllables - desired_syllables[i]))
        else:
            errors.append(obs_syllables)

    if len(lines) < len(desired_syllables):
        for i in range(len(lines),len(desired_syllables)):
            errors.append(desired_syllables[i])

    return errors


def total_syllable_error(limerick):

    lines = limerick.split('\n')
    #lines = re.split('<br.>', limerick)

    line_errors = line_syllable_error(lines)

    return sum(line_errors)

def average_line_syllable_error(limericks):

    total_line_errors = []
    line_counts = []
    for limerick in limericks:
        lim_lines = limerick.split('\n')
        #lim_lines = re.split('<br.>', limerick)

        line_errors = line_syllable_error(lines)

        # this does not assume that every poem has the same number of lines
        for i in range(len(line_errors)):
            if i >= len(total_line_errors):
                total_line_errors.append(line_error[i])
                line_counts.append(1.0)
            else:
                total_line_errors[i] += line_error[i]
                line_counts[i] += 1.0

    return [error / count for error, count in zip(total_line_errors, line_counts)]


def average_total_syllable_error(limericks):

    total_error = 0.0
    for limerick in limericks:
        total_error += total_syllable_error(limerick)

    return 1.0 * total_error / len(limericks)

def rhyming_error(limerick):

    global rhyming_line_sets
    
    lines = limerick.split('\n')
    #lines = re.split('<br.>', limerick)

    line_words = [line.split() for line in lines]

    total_rhyme_error = 0
    for rhyming_lines in rhyming_line_sets:

        min_error = float("inf")
        for i in rhyme_lines:

            rhymes = pro.rhymes(line_words[i][-1])
            rhyme_error = 0
            for j in (rhyme_lines[:i] + rhyme_lines[i+1:]):
                if line_words[j][-1] not in rhymes:
                    rhyme_error += 1

            if rhyme_error < min_error:
                min_error = rhyme_error
        total_rhyme_error += min_error

    return total_rhyme_error
