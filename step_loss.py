from matplotlib import pyplot as plt


def read_file(file_path):
    train_loss = []
    valid_loss = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.split(':')
            if line[0] == 'train_step_epoch':
                train_loss.append(float(line[1]))
            elif line[0] == 'valid_step_epoch':
                valid_loss.append(float(line[1]))
            elif line[0] == 'EPOCH':
                print(line)
    return train_loss, valid_loss


train_loss, valid_loss = read_file('LSTM 2 0.5 STEP LOSS.file')
plt.plot(valid_loss)
plt.plot(train_loss)
plt.show()

