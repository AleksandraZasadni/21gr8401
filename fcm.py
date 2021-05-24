import cv2
import numpy as np
import random
import sys
import nibabel as nib
import matplotlib.pyplot as plt
import glob
import os 
import datetime
from PIL import Image
import time
from datetime import timedelta


def init_mem_mat(nb_pixels, nb_clusters):
    mem_mat = np.zeros((nb_pixels, nb_clusters))
    x = np.arange(nb_pixels)
    for j in range(nb_clusters):
        xj = x % nb_clusters == j
        mem_mat[xj, j] = 1

    return mem_mat


def compute_centers(img_mat, mem_mat, fuzzy):
    num = np.dot(img_mat, mem_mat ** fuzzy)
    dem = np.sum(mem_mat ** fuzzy, axis=0)

    return num / dem


def update_mem_mat(ctr_mat, img_mat, fuzzy):
    ctr_mat_mesh, img_mat_mesh = np.meshgrid(ctr_mat, img_mat)
    power = 2. / (fuzzy - 1)
    p1 = abs(img_mat_mesh - ctr_mat_mesh) ** power
    p2 = np.sum((1. / abs(img_mat_mesh - ctr_mat_mesh)) ** power, axis=1)

    return 1. / (p1 * p2[:, None])


start_time = time.monotonic()


for k in range(0,511):
    img_pathTest = 'D:/ST8/data/single/split/test/p13_slice{0}.png'.format(k+1)
    img = cv2.imread(img_pathTest)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   

    # Number of clusters
    K = 4

    # Number of data/pixels
    N = gray.size

    # Fuzzyness coefficient
    m = 2

    # Threshold
    eps = 0.03

    # Maximum number of iterations
    max_i = 100

    # Initialization
    X = gray.flatten().astype('float')
    U = init_mem_mat(N, K)

    # Repeat until convergence
    i = 0
    while True:
        # Compute centroid for each cluster
        C = compute_centers(X, U, m)

        # Save initial membership matrix
        old_U = np.copy(U)

        # Update coefficients for each pixel
        U = update_mem_mat(C, X, m)

        # Difference between initial mem matrix and new one
        d = np.sum(abs(U - old_U))
        print(str(i) + " - d = " + str(d))

        # Check convergence
        if d < eps or i > max_i:
            break
        i += 1

    # Segmentation
    seg = np.argmax(U, axis=1)
    seg = seg.reshape(gray.shape).astype('int')


    plt.imshow(seg,  cmap="gray")
    plt.imsave('D:/ST8/data/single/split/test/Validation/FCM/all/all/FCM_p13_slice{0}.png'.format(k+1),seg,cmap='gray')
    plt.show()
    print('step:', k)
    
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

for k in range(0,511):
    img_pathTest = 'D:/ST8/data/single/split/test/p15_slice{0}.png'.format(k+1)
    img = cv2.imread(img_pathTest)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Number of clusters
    K = 4

    # Number of data/pixels
    N = gray.size

    # Fuzzyness coefficient
    m = 2

    # Threshold
    eps = 0.03

    # Maximum number of iterations
    max_i = 100

    # Initialization
    X = gray.flatten().astype('float')
    U = init_mem_mat(N, K)

    # Repeat until convergence
    i = 0
    while True:
        # Compute centroid for each cluster
        C = compute_centers(X, U, m)

        # Save initial membership matrix
        old_U = np.copy(U)

        # Update coefficients for each pixel
        U = update_mem_mat(C, X, m)

        # Difference between initial mem matrix and new one
        d = np.sum(abs(U - old_U))
        print(str(i) + " - d = " + str(d))

        # Check convergence
        if d < eps or i > max_i:
            break
        i += 1

    # Segmentation
    seg = np.argmax(U, axis=1)
    seg = seg.reshape(gray.shape).astype('int')


    plt.imshow(seg,  cmap="gray")
    plt.imsave('D:/ST8/data/single/split/test/Validation/FCM/all/all/FCM_p15_slice{0}.png'.format(k+1),seg,cmap='gray')
    plt.show()
    print('step:', k)
    
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

for k in range(0,511):
    img_pathTest = 'D:/ST8/data/single/split/test/p18_slice{0}.png'.format(k+1)
    img = cv2.imread(img_pathTest)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Number of clusters
    K = 4

    # Number of data/pixels
    N = gray.size

    # Fuzzyness coefficient
    m = 2

    # Threshold
    eps = 0.03

    # Maximum number of iterations
    max_i = 100

    # Initialization
    X = gray.flatten().astype('float')
    U = init_mem_mat(N, K)

    # Repeat until convergence
    i = 0
    while True:
        # Compute centroid for each cluster
        C = compute_centers(X, U, m)

        # Save initial membership matrix
        old_U = np.copy(U)

        # Update coefficients for each pixel
        U = update_mem_mat(C, X, m)

        # Difference between initial mem matrix and new one
        d = np.sum(abs(U - old_U))
        print(str(i) + " - d = " + str(d))

        # Check convergence
        if d < eps or i > max_i:
            break
        i += 1

    # Segmentation
    seg = np.argmax(U, axis=1)
    seg = seg.reshape(gray.shape).astype('int')


    plt.imshow(seg,  cmap="gray")
    plt.imsave('D:/ST8/data/single/split/test/Validation/FCM/all/all/FCM_p18_slice{0}.png'.format(k+1),seg,cmap='gray')
    plt.show()
    print('step:', k)
    
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

for k in range(0,511):
    img_pathTest = 'D:/ST8/data/single/split/test/p20_slice{0}.png'.format(k+1)
    img = cv2.imread(img_pathTest)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Number of clusters
    K = 4

    # Number of data/pixels
    N = gray.size

    # Fuzzyness coefficient
    m = 2

    # Threshold
    eps = 0.03

    # Maximum number of iterations
    max_i = 100

    # Initialization
    X = gray.flatten().astype('float')
    U = init_mem_mat(N, K)

    # Repeat until convergence
    i = 0
    while True:
        # Compute centroid for each cluster
        C = compute_centers(X, U, m)

        # Save initial membership matrix
        old_U = np.copy(U)

        # Update coefficients for each pixel
        U = update_mem_mat(C, X, m)

        # Difference between initial mem matrix and new one
        d = np.sum(abs(U - old_U))
        print(str(i) + " - d = " + str(d))

        # Check convergence
        if d < eps or i > max_i:
            break
        i += 1

    # Segmentation
    seg = np.argmax(U, axis=1)
    seg = seg.reshape(gray.shape).astype('int')


    plt.imshow(seg,  cmap="gray")
    plt.imsave('D:/ST8/data/single/split/test/Validation/FCM/all/all/FCM_p28_slice{0}.png'.format(k+1),seg,cmap='gray')
    plt.show()
    print('step:', k)
    
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

for k in range(0,511):
    img_pathTest = 'D:/ST8/data/single/split/test/p28_slice{0}.png'.format(k+1)
    img = cv2.imread(img_pathTest)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Number of clusters
    K = 4

    # Number of data/pixels
    N = gray.size

    # Fuzzyness coefficient
    m = 2

    # Threshold
    eps = 0.03

    # Maximum number of iterations
    max_i = 100

    # Initialization
    X = gray.flatten().astype('float')
    U = init_mem_mat(N, K)

    # Repeat until convergence
    i = 0
    while True:
        # Compute centroid for each cluster
        C = compute_centers(X, U, m)

        # Save initial membership matrix
        old_U = np.copy(U)

        # Update coefficients for each pixel
        U = update_mem_mat(C, X, m)

        # Difference between initial mem matrix and new one
        d = np.sum(abs(U - old_U))
        print(str(i) + " - d = " + str(d))

        # Check convergence
        if d < eps or i > max_i:
            break
        i += 1

    # Segmentation
    seg = np.argmax(U, axis=1)
    seg = seg.reshape(gray.shape).astype('int')


    plt.imshow(seg,  cmap="gray")
    plt.imsave('D:/ST8/data/single/split/test/Validation/FCM/all/all/FCM_p28_slice{0}.png'.format(k+1),seg,cmap='gray')
    plt.show()
    print('step:', k)
    
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

for k in range(0,511):
    img_pathTest = 'D:/ST8/data/single/split/test/p29_slice{0}.png'.format(k+1)
    img = cv2.imread(img_pathTest)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Number of clusters
    K = 4

    # Number of data/pixels
    N = gray.size

    # Fuzzyness coefficient
    m = 2

    # Threshold
    eps = 0.03

    # Maximum number of iterations
    max_i = 100

    # Initialization
    X = gray.flatten().astype('float')
    U = init_mem_mat(N, K)

    # Repeat until convergence
    i = 0
    while True:
        # Compute centroid for each cluster
        C = compute_centers(X, U, m)

        # Save initial membership matrix
        old_U = np.copy(U)

        # Update coefficients for each pixel
        U = update_mem_mat(C, X, m)

        # Difference between initial mem matrix and new one
        d = np.sum(abs(U - old_U))
        print(str(i) + " - d = " + str(d))

        # Check convergence
        if d < eps or i > max_i:
            break
        i += 1

    # Segmentation
    seg = np.argmax(U, axis=1)
    seg = seg.reshape(gray.shape).astype('int')


    plt.imshow(seg,  cmap="gray")
    plt.imsave('D:/ST8/data/single/split/test/Validation/FCM/all/all/FCM_p28_slice{0}.png'.format(k+1),seg,cmap='gray')
    plt.show()
    print('step:', k)
    
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
