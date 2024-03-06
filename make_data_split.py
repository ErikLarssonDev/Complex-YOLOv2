import random

data_dir = 'zod/ImageSets/'
# Open the train.txt file for reading
with open(data_dir + "train.txt", "r") as train_file:
    lines = train_file.readlines()

# Shuffle the lines randomly
random.shuffle(lines)

# Open the val.txt file for writing
with open(data_dir + "val.txt", "w") as val_file:
    # Write the first 10000 lines to val.txt
    val_file.writelines(lines[:10000])

# Remove the transferred lines from train.txt
with open(data_dir + "train.txt", "w") as train_file:
    # Write the remaining lines back to train.txt
    train_file.writelines(lines[10000:])