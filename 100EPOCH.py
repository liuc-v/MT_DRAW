import matplotlib.pyplot as plt

x = []
train_loss = []
valid_loss = []

# x.append(0)
# train_loss.append(8)
# valid_loss.append(8)

with open('100EPOCH.txt', 'r') as f:
    for line in f:
        epoch, train_loss_val, valid_loss_val = line.strip().split(', ')
        x.append(int(epoch.split(':')[-1]))
        train_loss.append(float(train_loss_val.split(':')[-1]))
        valid_loss.append(float(valid_loss_val.split(':')[-1]))



plt.plot(x, train_loss, label='train loss')
plt.plot(x, valid_loss, label='valid loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
