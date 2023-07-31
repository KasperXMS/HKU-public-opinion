import matplotlib.pyplot as plt
import pylab as pl
log_file = "/userhome/cs2/u3603202/nlp/project/output/roberta-1e-5/log.txt"

f = open(log_file)
loss = f.readlines()
result = []
for i in loss:
    if '|' not in i or 'E' in i:
        continue
    tmp = i.split("|")[2]
    tmp = float(tmp.strip())
    result.append(tmp)

print(result)
x = result
y = range(0, len(result))
fig = plt.figure(figsize = (7,5)) 
ax1 = fig.add_subplot(1, 1, 1)
pl.plot(y,x,'g-',label=u'RoBERTa')
pl.legend()
pl.xlabel(u'iters')
pl.ylabel(u'loss')
pl.title('RoBERTa loss in training')
plt.savefig("/userhome/cs2/u3603202/nlp/project/output/roberta-1e-5/train_results_loss_new.png")

