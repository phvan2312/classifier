import cPickle

neg_path = './negative-words.txt'
pos_path = './positive-words.txt'

lexicon_out_path = './lexicon.pkl'

if __name__ == "__main__":
    dct = {}

    for path,score in [(neg_path,'neg'),(pos_path,'pos')]:
        for line in open(path,'r'):
            dct[line.strip().lower()] = score

    # save
    cPickle.dump(dct,open(lexicon_out_path,'w'))