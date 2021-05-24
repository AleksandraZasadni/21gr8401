import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


def dice(imagePath1, imagePath2):
    reference_image = cv2.imread(imagePath1)
    reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    reference_image = np.interp(reference_image, [np.min(reference_image), np.max(reference_image)], [0, 1])
    reference_array = np.array(reference_image)
    result_image = cv2.imread(imagePath2)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    result_image = np.interp(result_image, [np.min(result_image), np.max(result_image)], [0, 1])
    result_array = np.array(result_image)
    im1 = np.asarray(result_array).astype(np.bool)
    im2 = np.asarray(reference_array).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def load_images(input_path):
    """
    Load images

    :param input_path: input path to png image
    :return: return a loaded gray-scaled images
    """
    img = cv2.imread(input_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def lesion_mask(original_image_path, segmented_image_path, threshold=230):
    """
    Lesion mask algorithm

    :param original_image_path: original_image_path: path to preprocesed image (original)
    :param consencus_path: path to cencensus image
    :param segmented_image_path: path to the segmented image
    :param threshold: intensity level that needed tko be surpassed before cluster is classified as lesion
    :return: binary image, where 0 = no lesion, 255 = lesion
    """
    # LOAD IMAGE
    img = load_images(original_image_path)
    img_seg = load_images(segmented_image_path)

    # SETUP
    uniqueInt = np.unique(img_seg)  # intensity value of segmented clusters
    shape = uniqueInt.shape  # number of segmented clusters
    cnts = np.zeros([shape[0], 1])  # create a counter for each cluster group

    # FIND CLUSTER CONTAINING LESION
    # Add to the cluster counter if cluster coordinate surpasses the threshold of 230 in original image
    for i in range(len(img_seg)):
        for j in range(len(img_seg[i])):
            if img[i, j] > threshold:  # original image surpasses 230
                for k in range(len(uniqueInt)):  # find which cluster group contains this coordinate
                    if uniqueInt[k] == img_seg[i, j]:
                        cnts[k] = cnts[k] + 1  # add an to this cluster groups counter

    maxElement = np.amax(cnts)  # maximum number of count
    indexPos = np.where(cnts == maxElement)  # Which cluster group had maximum
    classLesionNr = uniqueInt[indexPos[0]]  # cluster group what contains lesion.

    # CREATE BINARY MASK
    mask = np.zeros(img_seg.shape)  # create lesion mask
    if maxElement > 0 and len(
            uniqueInt) > 1:  # ensure found cluster group had any hits and itsnt the only clyster group
        for i in range(len(img_seg)):
            for j in range(len(img_seg[i])):
                if img_seg[i, j] == classLesionNr:
                    mask[i, j] = 255
    return mask


def plot_and_dice(segmented_image_path, consensus_path, lesion_mask: np.ndarray):
    # LOAD IMAGE
    img_seg = load_images(segmented_image_path)
    img_cons = load_images(consensus_path)

    plt.figure(figsize=(12, 8))
    plt.subplot(131)
    plt.imshow(img_seg, cmap='gray')
    plt.title('Segmented image')

    plt.subplot(132)
    plt.imshow(img_cons, cmap='gray')
    plt.title('Consensus image')

    plt.subplot(133)
    plt.imshow(lesion_mask, cmap="gray")
    plt.title("Lesion mask")
    plt.show()

    shape = img_seg.shape
    mask2 = np.zeros([shape[0], shape[1], 3])
    for i in range(len(img_seg)):
        for j in range(len(img_seg[i])):
            if lesion_mask[i, j] == img_cons[i, j] & img_cons[i, j] == 255:
                mask2[i, j, 1] = 1  # green - correct positiv
            if lesion_mask[i, j] > img_cons[i, j]:
                mask2[i, j, 2] = 1  # BLUE - false positiv
            if lesion_mask[i, j] < img_cons[i, j]:
                mask2[i, j, 0] = 1  # RED - false negativ
    plt.imshow(mask2, cmap="gray")
    plt.title("Final Result (blue=FP, red=FN)")
    plt.show()

    plt.imsave(os.path.join('C:/Users/aleks/Desktop/ST8-maske/tempImg', 'mask.png'), lesion_mask, cmap='gray')
    plt.imsave(os.path.join('C:/Users/aleks/Desktop/ST8-maske/tempImg', 'cons.png'), img_cons)

    img1_path = 'C:/Users/aleks/Desktop/ST8-maske/tempImg/mask.png'
    img2_path = 'C:/Users/aleks/Desktop/ST8-maske/tempImg/cons.png'

    return dice(img1_path, img2_path)

diceScoreListP13 = []
for p in range(0,511):
    img_path = 'D:/ST8/data/single/split/test/p13_slice{0}.png'.format(p+1)
    cons_path = 'D:/ST8/data/single/groundTruth/sliced/patient13/test/gt_p13_slice{0}.png'.format(p+1)
    seg_path = 'D:/ST8/data/single/uNet/Results/PT13_new/flipped/PT13_{0}_predicted.png'.format(p+1)

    
    mask = lesion_mask(original_image_path=img_path, segmented_image_path=seg_path)
    diceScore = plot_and_dice(segmented_image_path=seg_path, consensus_path=cons_path, lesion_mask=mask)
    print(diceScore)
    diceScoreListP13.append(diceScore)
    print('p',p)

AvgDiceP13 = sum(diceScoreListP13)/512
print('AvgDice', AvgDiceP13)

diceScoreListP15 = []
for p in range(0,511):
    img_path = 'D:/ST8/data/single/split/test/p15_slice{0}.png'.format(p+1)
    cons_path = 'D:/ST8/data/single/groundTruth/sliced/patient15/test/gt_p15_slice{0}.png'.format(p+1)
    seg_path = 'D:/ST8/data/single/uNet/Results/PT15_new/flipped/PT15_{0}_predicted.png'.format(p+1)

    
    mask = lesion_mask(original_image_path=img_path, segmented_image_path=seg_path)
    diceScore = plot_and_dice(segmented_image_path=seg_path, consensus_path=cons_path, lesion_mask=mask)
    print(diceScore)
    diceScoreListP15.append(diceScore)
    print('p',p)

AvgDiceP15 = sum(diceScoreListP15)/512
print('AvgDice', AvgDiceP15)

diceScoreListP18 = []
for p in range(0,511):
    img_path = 'D:/ST8/data/single/split/test/p18_slice{0}.png'.format(p+1)
    cons_path = 'D:/ST8/data/single/groundTruth/sliced/patient18/test/gt_p18_slice{0}.png'.format(p+1)
    seg_path = 'D:/ST8/data/single/uNet/Results/PT18_new/flipped/PT18_{0}_predicted.png'.format(p+1)

    
    mask = lesion_mask(original_image_path=img_path, segmented_image_path=seg_path)
    diceScore = plot_and_dice(segmented_image_path=seg_path, consensus_path=cons_path, lesion_mask=mask)
    print(diceScore)
    diceScoreListP18.append(diceScore)
    print('p',p)

AvgDiceP18 = sum(diceScoreListP18)/512
print('AvgDice', AvgDiceP18)

diceScoreListP20 = []
for p in range(0,511):
    img_path = 'D:/ST8/data/single/split/test/p20_slice{0}.png'.format(p+1)
    cons_path = 'D:/ST8/data/single/groundTruth/sliced/patient20/test/gt_p20_slice{0}.png'.format(p+1)
    seg_path = 'D:/ST8/data/single/uNet/Results/PT20_new/flipped/PT20_{0}_predicted.png'.format(p+1)

    
    mask = lesion_mask(original_image_path=img_path, segmented_image_path=seg_path)
    diceScore = plot_and_dice(segmented_image_path=seg_path, consensus_path=cons_path, lesion_mask=mask)
    print(diceScore)
    diceScoreListP20.append(diceScore)
    print('p',p)

AvgDiceP20 = sum(diceScoreListP20)/512
print('AvgDice', AvgDiceP20)

diceScoreListP28 = []
for p in range(0,511):
    img_path = 'D:/ST8/data/single/split/test/p28_slice{0}.png'.format(p+1)
    cons_path = 'D:/ST8/data/single/groundTruth/sliced/patient28/test/gt_p28_slice{0}.png'.format(p+1)
    seg_path = 'D:/ST8/data/single/uNet/Results/PT28_new/flipped/PT28_{0}_predicted.png'.format(p+1)

    
    mask = lesion_mask(original_image_path=img_path, segmented_image_path=seg_path)
    diceScore = plot_and_dice(segmented_image_path=seg_path, consensus_path=cons_path, lesion_mask=mask)
    print(diceScore)
    diceScoreListP28.append(diceScore)
    print('p',p)

AvgDiceP28 = sum(diceScoreListP28)/512
print('AvgDice', AvgDiceP28)

diceScoreListP29 = []
for p in range(0,511):
    img_path = 'D:/ST8/data/single/split/test/p29_slice{0}.png'.format(p+1)
    cons_path = 'D:/ST8/data/single/groundTruth/sliced/patient29/test/gt_p29_slice{0}.png'.format(p+1)
    seg_path = 'D:/ST8/data/single/uNet/Results/PT29_new/flipped/PT29_{0}_predicted.png'.format(p+1)

    
    mask = lesion_mask(original_image_path=img_path, segmented_image_path=seg_path)
    diceScore = plot_and_dice(segmented_image_path=seg_path, consensus_path=cons_path, lesion_mask=mask)
    print(diceScore)
    diceScoreListP29.append(diceScore)
    print('p',p)

AvgDiceP29 = sum(diceScoreListP29)/512
print('AvgDice', AvgDiceP29)
