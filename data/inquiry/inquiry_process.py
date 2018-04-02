from sklearn.model_selection import train_test_split

inquiry_data_path = './all.txt'
inquiry_train_data_out = './train.txt'
inquiry_test_data_out  = './test.txt'

def read_data(data_path, test_percent):
    words, labels = [], []

    for line in open(data_path):
        print line
        datas = line.strip().split('\t')

        if len(datas) < 2:
            continue

        label = datas[0].strip()
        text  = datas[1].strip()

        words.append(text)
        labels.append(label)

    train_words, train_labels, test_words, test_labels = split_batch(words, labels, test_percent)

    def summarize(x_words,x_labels):
        results = []
        for x_label, x_word in zip(x_words,x_labels):
            results.append("%s\t%s" % (x_word,x_label))

        return results

    train_data = summarize(train_words,train_labels)
    test_data  = summarize(test_words, test_labels)

    return train_data, test_data

def split_batch(words, labels, test_percent):
    train_words, test_words, train_labels, test_labels = train_test_split(words,labels,test_size=test_percent,random_state=43,stratify=labels)

    return train_words, train_labels, test_words, test_labels

if __name__ == '__main__':
    train_data, test_data = read_data(inquiry_data_path,0.4)

    with open(inquiry_train_data_out,'w') as f:
        f.write('\n'.join(train_data))

    with open(inquiry_test_data_out,'w') as f:
        f.write('\n'.join(test_data))

    pass