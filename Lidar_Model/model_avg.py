import numpy as np

from M6_AutoEncoder_LSTM import run_task

cross_tb = np.zeros((5,5))
total_acc = 0
t = 1
def ct2np(ct):
    temp = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            temp[i][j] = ct.loc[i,j]
    return(temp)

for i in range(t):
    (ct, pre_acc) = run_task()
    total_acc+=pre_acc
    # print(pre_acc)
    temp = ct2np(ct)
    cross_tb+=temp
    print("\n\n\n\n\n")
    print(i+1)
    print(cross_tb / (i+1))
    print("The avg acc => %f" % (total_acc / (i+1)))

print("\n\n\n\n\n")
print(cross_tb/t)
print("The avg acc => %f" % (total_acc/t))

