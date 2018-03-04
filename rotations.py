#! python3
import os
import pickle
import sys
import time
from pprint import pformat
from shutil import copyfile
from mean_average_precision import compute_map
from mean_average_precision import compute_map_fast
from utils.random_rotation import random_rotation
from random import random
import threading
import numpy as np

from evaluate_performance import evaluate

import bimpy


def GetBaseRotation(alpha, size):
    alphas = np.sin(alpha)
    alphac = np.cos(alpha)
    flat_rotation = np.array([[alphac, -alphas], [alphas, alphac]])
    I = np.eye(size)
    I[0:2, 0:2] = flat_rotation
    return I.astype(np.float32)


ctx = bimpy.Context()

ctx.init(1200, 1200, "ITQ")

with ctx:
    bimpy.themes.set_light_theme()

opened = bimpy.Bool(True)

HASH_SIZE = 2
DATA_POINTS = bimpy.Int(200)
CLASSES = bimpy.Int(4)
MAP_SCORE = 0

std = bimpy.Float(0.2)

colors = [0x4b19e6, 0x4bb43c, 0x19e1ff, 0xc88200, 0x3182f5, 0xb41e91, 0xf0f046, 0xf032e6, 0xd2f53c,
          0xfabebe, 0x008080, 0xe6beff, 0xaa6e28, 0xfffac8, 0x800000, 0xaaffc3, 0x808000, 0xffd8b1,
          0x000080, 0x808080, 0xFFFFFF, 0x000000]


def normalize(x):
    norm = np.linalg.norm(x, axis=1)
    norm = np.expand_dims(norm, axis=1)
    x /= norm

with open('H.pkl', 'rb') as pkl:
    H = pickle.load(pkl)
with open('labels.pkl', 'rb') as pkl:
    labels = pickle.load(pkl)
with open('H_test.pkl', 'rb') as pkl:
    H_test = pickle.load(pkl)
with open('labels_test.pkl', 'rb') as pkl:
    labels_test = pickle.load(pkl)

H = H.astype(np.float32)
H_test = H_test.astype(np.float32)

normalize(H)
normalize(H_test)


CLASSES = 10
HASH_SIZE = 48
DATA_POINTS = 1000
std = 0.2

def generate_fake_data():
    classes = np.random.rand(CLASSES, HASH_SIZE) - 0.5
    for k in range(5):
        for i in range(CLASSES):
            for j in range(CLASSES):
                if i != j:
                    c = classes[j, :] - classes[i, :]
                    l = np.linalg.norm(c)
                    c /= l
                    classes[i, :] -= c * (2.0 - l) * 0.1
        normalize(classes)
    labels = np.random.randint(CLASSES, size=DATA_POINTS)
    data = classes[labels, :] + np.random.normal(scale=std, size=(DATA_POINTS, HASH_SIZE))
    normalize(data)
    print('hash_size %d' % data.shape[1])

    for i in range(labels.size):
        labels[i] = 1 << labels[i]

    return labels, data

#labels, H = generate_fake_data()
#labels_test = labels
#H_test = H

size = labels.shape[0]

normalize(H)
normalize(H_test)

b_train_o = H
l_train_o = labels
b_test_o = H_test
l_test_o = labels_test

#if size > 25000:
idx = np.random.randint(size, size=2000)
size = 2000
labels_original = labels
labels = labels[idx, :]
H_original = H
H = H[idx, :]

#H = H[:, :2]
#H_test = H_test[:, :2]

normalize(H)
normalize(H_test)


b_train = H
l_train = labels
b_test = H_test
l_test = labels_test


R = np.eye(2, 2, dtype=np.float32)
IR = np.eye(2, 2, dtype=np.float32)
axis = x = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


def doRotation():
    global R
    global IR
    global b_train
    global l_train
    global b_test
    global l_test
    global labels
    global size
    global H
    
    if True:
        S = np.bitwise_and(np.reshape(labels, [size, 1]),
                           np.reshape(labels, [1, size])).astype(dtype=np.bool)
    else:
        S = np.equal(np.reshape(labels, [size, 1]), np.reshape(labels, [1, size]))

    S = S * 2.0 - 1.0

    hash_size = H.shape[1]

    R = np.eye(hash_size, hash_size, dtype=np.float32)

    for i in range(500):
        #update B
        #B = np.matmul(S, np.sign(np.matmul(H, R)))
        B = np.sign(np.matmul(H, R))

        #update R
        U, s, Vh = np.linalg.svd(np.matmul(B.T, H), full_matrices=False)
        R = np.matmul(U, Vh)
        #R = np.matmul(Vh.T, U.T)


    R_ = np.eye(hash_size, hash_size, dtype=np.float32)
    #
    # tr = np.trace(np.matmul(H.T, np.matmul(S, np.sign(H))))
    # for i in range(1000):
    #     random_R = random_rotation.random_rotation(hash_size)
    #     VR = np.matmul(H, random_R)
    #     h = np.sign(VR)
    #     B = np.matmul(VR.T, np.matmul(S, h))
    #     tr_ = np.trace(B)
    #     if tr_ > tr:
    #         tr = tr_
    #         R_ = random_R

    R_ = np.eye(hash_size, hash_size, dtype=np.float32)
    for i in range(550):
        #update B
        B = np.matmul(S, np.sign(np.matmul(H, R_)))
        #B = np.sign(np.matmul(H, R))

        #update R
        U, s, Vh = np.linalg.svd(np.matmul(B.T, H), full_matrices=False)
        R_ = np.matmul(U, Vh)
        #R_ = np.matmul(Vh.T, U.T)

    l_db = l_train
    b_db = b_train

    _, map_test, _ = evaluate(
          l_train
        , b_train
        , l_train
        , b_train
        , l_db
        , b_db
        , top_n=0
        , and_mode=True
        , force_slow=True
        , testOnTrain=False)

    print("Test on train: %f" % map_test)

    _, map_test, _ = evaluate(
          l_train
        , b_train
        , l_test
        , b_test
        , l_db
        , b_db
        , top_n=0
        , and_mode=True
        , force_slow=True
        , testOnTrain=False)

    print("Test on test: %f" % map_test)

    b_train_r = np.matmul(b_train, R)
    b_test_r = np.matmul(b_test, R)
    #b_db_r = np.matmul(self.b_db, R)
    l_db = l_train
    b_db_r = b_train_r


    print("After rotation:")

    _, map_test, _ = evaluate(
          l_train
        , b_train_r
        , l_train
        , b_train_r
        , l_db
        , b_db_r
        , top_n=0
        , and_mode=True
        , force_slow=True
        , testOnTrain=False)

    print("Test on train: %f" % map_test)

    _, map_test, _ = evaluate(
          l_train
        , b_train_r
        , l_test
        , b_test_r
        , l_db
        , b_db_r
        , top_n=0
        , and_mode=True
        , force_slow=True
        , testOnTrain=False)

    print("Test on test: %f" % map_test)

    b_train_r = np.matmul(b_train, R_)
    b_test_r = np.matmul(b_test, R_)
    #b_db_r = np.matmul(self.b_db, R)
    l_db = l_train
    b_db_r = b_train_r

    print("After rotation2:")

    _, map_test, _ = evaluate(
          l_train
        , b_train_r
        , l_train
        , b_train_r
        , l_db
        , b_db_r
        , top_n=0
        , and_mode=True
        , force_slow=True
        , testOnTrain=False)

    print("Test on train: %f" % map_test)

    _, map_test, _ = evaluate(
          l_train
        , b_train_r
        , l_test
        , b_test_r
        , l_db
        , b_db_r
        , top_n=0
        , and_mode=True
        , force_slow=True
        , testOnTrain=False)

    print("Test on test: %f" % map_test)

    print("Start")
    print("==========================================")

    idx = np.random.permutation(labels_original.shape[0])
    idx_q = np.copy(idx[:1000])
    idx_db = np.copy(idx[1000:][:16000])
    labels_q = labels_original[idx_q, :]
    H_q =  H_original[idx_q, :]
    labels_db = labels_original[idx_db, :]
    H_db =  H_original[idx_db, :]

    mapd0, _ = compute_map(H_db, H_q, labels_db, labels_q, and_mode=True, force_slow=False)
    step = 1.0
    R = R.astype(np.float32)

    worker_count = 1
    steps = int(800 / worker_count)
    results = [(0, np.eye(hash_size, hash_size, dtype=np.float32)) for i in range(worker_count)]

    for i in range(steps):
        step = (steps - i) / steps

        def ComputeNewValue(w):
            rBasis = random_rotation(HASH_SIZE).astype(np.float32)
            if random() > 0.5:
                s = step
            else:
                s = -step

            deltaR = np.matmul(rBasis.T, np.matmul(GetBaseRotation(s, HASH_SIZE), rBasis))
            newR = np.matmul(R, deltaR)
            rotated_data = np.matmul(H_db, newR)
            rotated_data_q = np.matmul(H_q, newR)
            mapd1 = compute_map_fast(rotated_data, rotated_data_q, labels_db, labels_q, and_mode=True)
            results[w] = (mapd1, newR)

        threads = []
        for w in range(worker_count):
            t = threading.Thread(target=ComputeNewValue, args=(w, ))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        updated = False
        for w in range(worker_count):
            print("%f " % results[w][0], end='')
            if results[w][0] > mapd0:
                R = results[w][1]
                mapd0 = results[w][0]
                updated = True
        print("")
        if updated:
            print("++++++++++++++ %f ++++++++++++++++" % mapd0)

    b_train = b_train_o
    l_train = l_train_o
    b_test = b_test_o
    l_test = l_test_o

    b_train_r = np.matmul(b_train, R)
    b_test_r = np.matmul(b_test, R)
    #b_db_r = np.matmul(self.b_db, R)
    l_db = l_train
    b_db_r = b_train_r

    print("Brute force rotation:")

    _, map_test, _ = evaluate(
          l_train
        , b_train_r
        , l_test
        , b_test_r
        , l_db
        , b_db_r
        , top_n=50000
        , and_mode=True
        , force_slow=True
        , testOnTrain=False)

    print("Test on test: %f" % map_test)


    print("No rotation:")

    b_train = b_train_o
    l_train = l_train_o
    b_test = b_test_o
    l_test = l_test_o
    l_db = l_train
    b_db = b_train

    _, map_test, _ = evaluate(
          l_train
        , b_train
        , l_test
        , b_test
        , l_db
        , b_db
        , top_n=50000
        , and_mode=True
        , force_slow=True
        , testOnTrain=False)

    print("Test on test: %f" % map_test)



while not ctx.should_close():
    ctx.new_frame()

    bimpy.begin("Data", flags=bimpy.WindowFlags.ShowBorders)

    window_pos = bimpy.get_window_pos()

    radius = 200.0
    center = bimpy.Vec2(50, 70) + window_pos + bimpy.Vec2(radius, radius)

    bimpy.add_circle(center, radius, 0x70000000, 100)
    #
    # for i in range(labels.size):
    #     point = bimpy.Vec2(H[i, 0], H[i, 1])
    #     bimpy.add_circle_filled(point * radius + center, 8, 0xAF000000 + colors[0], 100)
    #
    # axis_ = np.matmul(axis, IR)
    #
    # bimpy.add_line(
    #     center - bimpy.Vec2(axis_[0, 0], axis_[0, 1]) * radius,
    #     center - bimpy.Vec2(axis_[1, 0], axis_[1, 1]) * radius,
    #     0xFFFF0000, 1)
    #
    # bimpy.add_line(
    #     center - bimpy.Vec2(axis_[2, 0], axis_[2, 1]) * radius,
    #     center - bimpy.Vec2(axis_[3, 0], axis_[3, 1]) * radius,
    #     0xFFFF0000, 1)

    bimpy.end()

    bimpy.begin("Controls", flags=bimpy.WindowFlags.ShowBorders)

    if bimpy.button("ITQ"):
        doRotation()
        #IR = np.linalg.inv(R)

    bimpy.set_window_font_scale(3.0)
    bimpy.text("MAP: %f" % MAP_SCORE)
    bimpy.set_window_font_scale(2.0)

    bimpy.end()

    ctx.render()