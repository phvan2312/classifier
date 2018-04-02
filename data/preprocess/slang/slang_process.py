import cPickle

slang_path = './slangDict.txt'
slang_out_path = './slang.pkl'

if __name__ == "__main__":
    dct = {}

    for line in open(slang_path,'r'):
        datas = line.strip().split('-')

        slang,correct_word = datas[0].strip().lower(), datas[1].strip().lower()

        print slang,correct_word
        dct[slang] = correct_word

    # save
    cPickle.dump(dct,open(slang_out_path,'w'))