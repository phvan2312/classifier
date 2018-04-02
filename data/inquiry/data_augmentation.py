def augment(file_path):
    result = []

    for line in open(file_path,'r'):
        result.append(line.strip())

        datas = line.strip().split('\t')

        if datas[0] == '__label__inquiry':
            new_sample = datas[1]
            if '?' in new_sample:
                pass


    pass

if __name__ == '__main__':

    pass