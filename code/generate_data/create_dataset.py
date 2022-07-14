import os
import random
import numpy as np
from medpy.io import load

import tensorflow as tf

# ---------------------------------------------------------------------------------------------------------------------
data_path = "../../data"
img_size = 256
X_ = []
y_ = []

for root, dirs, files in os.walk(data_path):
    if len(files) != 0:
        for name in files:
            if name.endswith((".nii.gz")):
                if os.path.basename(root) != 'goodmask':
                    mri, _ = load(os.path.join(root, name))
                    imageDim_orig = np.shape(mri)
                    if np.shape(imageDim_orig)[0] == 3:
                        if imageDim_orig[0] != 256 or imageDim_orig[1] != 256:
                            mri = tf.image.resize(mri, [256, 256])
                        for idx in range(mri.shape[2]):
                            X_.append([mri[:, :, idx]])
                    else:
                        pass
                else:
                    mask, _ = load(os.path.join(root, name))
                    imageDim_orig = np.shape(mask)
                    if np.shape(imageDim_orig)[0] == 3:
                        if imageDim_orig[0] != 256 or imageDim_orig[1] != 256:
                            mask = tf.image.resize(mask, [256, 256])
                        for idx in range(mask.shape[2]):
                            y_.append([mask[:, :, idx]])
                    else:
                        pass
            else:
                pass


# ---------------------------------------------------------------------------------------------------------------------
training_data = list(zip(X_, y_))
random.shuffle(training_data)
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1)
y = np.array(y).reshape(-1, img_size, img_size, 1)

# ---------------------------------------------------------------------------------------------------------------------
np.save('../../data/X_train.npy', X)
np.save('../../data/y_train', y)