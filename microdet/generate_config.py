# (Loss: {dice, cross entropy, focal}, LR = {10e-i, i belongs to 1 to 5}, batch_size={8,16,32}, train={finetuning, only training head(no finetuning)})

# Generate 1 configuration.txt files of all possible combinations, use system args to run training_model.py

import os

loss = ['dice', 'cross_entropy', 'focal']
LR = [0.1 / (10 ** i) for i in range(5)]
batch_size = [8, 16, 32]
train_mode = [True, False]

with open('configuration.txt', 'w') as f:
    for i in loss:
        for j in LR:
            for m in batch_size:
                for n in train_mode:
                    f.writelines(f'{str(i)},{str(j)},{str(m)},{str(n)}\n')
                    
    f.close()