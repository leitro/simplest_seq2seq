from generateData import RATIO
GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
input_max_len = 15
output_max_len = 14

vocab = {}
with open('vocab.dat') as f:
    for idx, line in enumerate(f):
        vocab[line.strip()] = idx

vocab_size = len(vocab)
index2word = {v: k for k, v in vocab.items()}


def processData(input_file, output_file, ratio=RATIO):
    in_data = [] # input
    with open(input_file) as in_f:
        for l in in_f:
            tmp = [vocab.get(token, UNK_TOKEN) for token in l.strip().split(' ')]
            for _ in range(input_max_len - len(tmp)):
                tmp.append(END_TOKEN)
            in_data.append(tmp)

    out_data = [] # output
    with open(output_file) as out_f:
        for l in out_f:
            tmp = [GO_TOKEN]
            tmp.extend([vocab.get(token, UNK_TOKEN) for token in l.strip().split(' ')])
            for _ in range(output_max_len - len(tmp)):
                tmp.append(END_TOKEN)
            out_data.append(tmp)

    la_data = [] # label
    for l in out_data:
        la_data.append(l[1:] + [END_TOKEN])

    total_len = len(in_data)
    th = int(total_len * ratio)
    return (in_data[:th], la_data[:th], out_data[:th]), (in_data[th:], la_data[th:], out_data[th:])

if __name__ == '__main__':
    trainData, testData = processData('input.dat', 'output.dat')
