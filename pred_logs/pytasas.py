import subprocess as sub

f_cer = open('test_cer_tasas.log', 'w')

for i in range(100):
    gt = 'groundtruth.dat'
    decoded = 'test_predict_seq.'+str(i)+'.log'
    res_cer = sub.Popen(['./tasas_cer.sh', gt, decoded], stdout=sub.PIPE)
    res_cer = res_cer.stdout.read().decode('utf8')
    res_cer = float(res_cer)/100
    f_cer.write(str(res_cer))
    f_cer.write(' ')
    print(i)

f_cer.close()
