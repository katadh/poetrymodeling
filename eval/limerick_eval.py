import pronouncing as pro

desired_syllables = [9, 9, 6, 6, 9]

def line_syllables(line):

    phones = [pro.phones_for_word(word)[0] for word in line.split()]
    num_syllables = sum([pro.syllable_count(phone_seq) for phone_seq in phones])
    return num_syllables

def line_syllable_error(lines):

    global desired_syllables

    errors = []
    for line, d_syb in zip(lines, desired_syllables):
        obs_syllables = line_syllables(line)
        errors.append(abs(obs_syllables - d_syb))

    return errors


def total_syllable_error(limerick):

    lines = limerick.split('\n')

    line_errors = line_syllable_error(lines)
