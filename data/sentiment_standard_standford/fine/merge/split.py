from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    in_path = './merge.txt'

    out_train = './semEval_train.txt'
    out_test  = './semEval_test.txt'

    samples = []
    labels  = []

    for line in open(in_path,'r'):
        data = line.strip().split('\t')

        sample = data[3].strip()
        label  = int(data[2])

        if label == -2:
            label = '__label__very_negative'
        elif label == -1:
            label = '__label__negative'
        elif label == 0:
            label = '__label__neutral'
        elif label == 1:
            label = '__label__positive'
        elif label == 2:
            label = '__label__very_positive'

        samples.append(sample)
        labels.append(label)

    train_words, test_words, train_labels, test_labels = train_test_split(samples, labels, test_size=0.2,
                                                                          random_state=43, stratify=labels)

    with open(out_train,'w') as f:
        results = []

        for train_word, train_label in zip(train_words, train_labels):
            results.append("%s\t%s" % (train_label,train_word))

        f.write('\n'.join(results))

    with open(out_test, 'w') as f:
        results = []

        for test_word, test_label in zip(test_words, test_labels):
            results.append("%s\t%s" % (test_label, test_word))

        f.write('\n'.join(results))