import random
import string
import os

examples = 10000
symbols = 26 # a-z
length = [4, 10] # in the range of 4 to 10
RATIO = 0.9 # <size of training dataset> / <size of total dataset>


if __name__ == '__main__':
    with open('vocab.dat', 'w') as f:
        f.write("<SOS>\n<EOS>\n<UNK>\n")
        for i in list(string.ascii_lowercase):
            f.write(i+'\n')
    print('DONE! vocab.dat')

    with open('input.dat', 'w') as fin:
        with open('output.dat', 'w') as fout:
            for i in range(examples):
                inp = [random.choice(list(string.ascii_lowercase)) for _ in range(random.randrange(length[0], length[1]))]
                out = [chr(ord(x) + 2) for x in inp if ord(x) <= ord('x')]
                fin.write(' '.join([x for x in inp]) + '\n')
                fout.write(' '.join([x for x in out]) + '\n')
    print('DONE! input.dat')
    print('DONE! output.dat')

    if not os.path.exists('pred_logs'):
        os.makedirs('pred_logs')
    with open('pred_logs/groundtruth.dat', 'w') as fw:
        with open('output.dat', 'r') as fr:
            num = 0
            data = fr.readlines()
            size = int(len(data) * RATIO)
            for d in data[size:]:
                fw.write(str(num)+' ')
                num += 1
                fw.write(d)
    print('DONE! pred_logs/groundtruth.dat')
