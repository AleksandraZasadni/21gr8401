import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import time
from datetime import timedelta

plt.style.use("ggplot")

import cv2
from tensorflow.keras.models import Model, load_model, save_model

# %%

# Set Parameters
im_width = 256
im_height = 256

model = load_model('C:/Users/krist/Desktop/uNet/FinalDataSet/unet_seg.hdf5', custom_objects={'dice_coef_loss': "dice_coef_loss", 'iou': "iou", 'dice_coef': "dice_coef"})

OriMaskPath = 'C:/Users/krist/Desktop/uNet/FinalDataSet/OriginalMask/OriginalMask/'
save_path = 'C:/Users/krist/Desktop/uNet/FinalDataSet/Results/PT29_new/RGB'
TEST_PATH = 'C:/Users/krist/Desktop/uNet/FinalDataSet/Test/'

start_time = time.monotonic()
for i in range(512):
    img = io.imread(os.path.join(TEST_PATH, "PT29_" + "%d.tif" % i))
    img = cv2.resize(img, (im_height, im_width))
    img = img / 255
    img = img[np.newaxis, :, :, :]
    # start = time.perf_counter()
    pred = model.predict(img)
    prediction = np.squeeze(pred) > .5
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))






