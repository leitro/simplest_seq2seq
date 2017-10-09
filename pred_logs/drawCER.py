import matplotlib.pyplot as plt

cer = open('test_cer_tasas.log', 'r')
cer_data = cer.read().split(' ')[:-1]
cerr = [float(i) for i in cer_data]

plt.plot(cerr, 'r-')
cer_spot, = plt.plot(cerr, 'ro')
plt.legend([cer_spot], ['tasas CER'])
plt.xlabel('epoch')
plt.ylim(0, 1.1)
plt.title('test data character error rate')
plt.show()
cer.close()
