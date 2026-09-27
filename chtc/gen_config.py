# (Loss: {dice, cross entropy, focal}, LR = {10e-i, i belongs to 1 to 5}, batch_size={8,16,32}, train={finetuning, only training head(no finetuning)})

# Generate 1 configuration.txt files of all possible combinations, use system args to run training_model.py

import os

LR = [1e-3, 1e-4, 1e-5]
batch_size = [8, 16, 32]

with open('configuration.txt', 'w') as f:
    for i in LR:
        for j in batch_size:
            f.writelines(f'{str(i)},{str(j)}\n')
                
    f.close()