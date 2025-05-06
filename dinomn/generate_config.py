# (Loss: {dice, cross entropy, focal}, LR = {10e-i, i belongs to 1 to 5}, batch_size={8,16,32}, train={finetuning, only training head(no finetuning)})

# Generate 1 configuration.txt files of all possible combinations, use system args to run training_model.py

import os

# loss = ['dice', 'focal', 'combined']
# LR = [0.1 / (10 ** i) for i in range(5)]
LR = [1e-3, 1e-4, 1e-5] # smaller batch size should have smaller LR, maximum: 1e-5, the one used in training so far
batch_size = [4,8,12,16]
# finetune = [True]
# train_mode = [True, False]

with open('configuration.txt', 'w') as f:
    # for i in loss:
        for i in LR:
            for j in batch_size:
                f.writelines(f'{float(i)},{int(j)}\n')
                # for n in finetune:
                # f.writelines(f'{str(i)},{str(j)},{str(m)},{str(n)}\n')
                    
f.close()