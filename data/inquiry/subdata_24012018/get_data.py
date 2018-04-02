import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import json

def load_json(file_path,key_name):
    with open(file_path,'r') as f:
        result = json.load(f)

    return [sample[key_name].strip() for sample in result['result']]

def load_txt(file_path):
    result = []

    for line in open(file_path,'r'):
        result.append(line.strip())

    return result

if __name__ == '__main__':
    data1 = load_txt('./data.txt')
    data2 = load_txt('./HotelBooking.txt')
    data3 = load_json('./HotelBooking_check_inquiry23_01_2018_10_15_38.txt','text')
    data4 = load_json('./RealEstate_check_inquiry23_01_2018_1_31_26.txt','text')

    final = list(set(data1 + data2 + data3 + data4))
    final.remove("")

    samples = []
    for line in final:
        samples.append("%s\t%s" % ('__label__inquiry',line))

    with open('./all_data.txt','w') as f:
        f.write('\n'.join(samples))
