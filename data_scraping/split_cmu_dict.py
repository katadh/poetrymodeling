import random

def load_dict(path):

    cmu_dict = {}
    with open(path) as dict_file:
        for line in dict_file.readlines():
            if line.startswith(';;;'):
                continue
            else:
                key, value = line.split(' ', 1)
                cmu_dict[key] = value.strip()

    return cmu_dict
        
# mutates the original dictionary
# the resulting number of dicts will be one
# greater than the number of elements in dataset_sizes
# because we append on the remaning elements as the final set
def split_data(dataset_sizes, orig_dict):
    split_data = []
    for data_size in dataset_sizes:
        new_dict = {}
        for i in range(data_size):
            key = random.choice(orig_dict.keys())
            new_dict[key] = orig_dict[key]
            del orig_dict[key]
        split_data.append(new_dict)

    split_data.append(orig_dict)
    return split_data

def write_dicts(dicts, path):
    for i in range(len(dicts)):
        write_dict(dicts[i], path + '_' + str(i))

def write_dict(in_dict, path):
    with open(path, 'w+') as out_file:
        for key, value in in_dict.iteritems():
            out_file.write(key + " " + value + "\n")

CMU_DICT = "cmudict-0.7b_cleaned"
cmuDict = load_dict(CMU_DICT)
dicts = split_data([int(len(cmuDict)*0.8), int(len(cmuDict)*0.1), int(len(cmuDict)*0.1)], cmuDict)
write_dicts(dicts, "/Users/chaitanya/Desktop/Study/10701/poetrymodeling/data_scraping/final_cmu_dict")