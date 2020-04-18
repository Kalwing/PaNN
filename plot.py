import numpy as np
import matplotlib.pyplot as plt
import os


# TODO: Generate training loss plot and val loss/DSC
fig, ax = plt.subplots((1, 2), figsize=(15, 7.5))
for path, label in paths_labels:
    try:
        npypath = os.path.join(path, PATH, SUPERVISION_LVL, F'{"val_dice" if VAL else "tra_loss"}.npy')
        dice = np.load(npypath)
    except FileNotFoundError:
        continue
    curve = np.mean(dice, axis=1)[:,1] if VAL else np.mean(dice, axis=1)
    plt.plot(range(dice.shape[0]), curve, label=label)
    print(dice.shape)



ax.set_xlabel("Epochs")
ax.set_ylabel("Validation Dice" if VAL else "Training loss")
ax.legend()
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.savefig(F"merged/{PATH.split('/')[-1]}_{SUPERVISION_LVL}_{'val_dice' if VAL else 'tra_loss'}_curves.png")
plt.rcParams.update({'font.size': 30})
plt.show()