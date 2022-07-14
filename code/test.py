import os
import numpy as np
from medpy.io import load, save

import tensorflow as tf

# ---------------------------------------------------------------------------------------------------------------------

def test(input_path, model_path):
    model = tf.keras.models.load_model(model_path, compile=False)

    image_data, image_header = load(input_path)  # Load data
    image_data = np.moveaxis(image_data, -1, 0)  # Bring the last dim to the first
    input_data = image_data[..., np.newaxis]  # Add one axis to the end

    input_data = np.divide(input_data.astype(float), np.std(input_data.astype(float), axis=0),
                           out=np.zeros_like(input_data.astype(float)),
                           where=np.std(input_data.astype(float), axis=0) != 0)

    mask = model.predict(input_data, batch_size=1)
    mask = np.argmax(np.asarray(mask), axis=3).astype(float)  # Find mask
    mask = np.moveaxis(mask, 0, -1)  # Bring the first dim to the last

    save(mask, os.path.dirname(input_path)+'/mask_'+os.path.basename(input_path), image_header)

# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Hyperparamaters """
    model_path = "../results/unet.model" #path to model
    input_path = '../data/train/test_normal/Normal01/fetus_04.nii' # path to data

    """ Predict """
    test(input_path, model_path)