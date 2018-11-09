#!/usr/bin/env python

from liblvq import lvq

import sys


classifier = lvq(3, 6)


train_set = (
    ((1, 0, 0), 0),
    ((0, 1, 0), 1),
    ((0, 0, 1), 2),
    ((1, 1, 0), 3),
    ((1, 0, 1), 4),
    ((1, 1, 1), 5),

    (( 0.8,  0.1, -0.2), 0),
    (( 0.2,  1.1, -0.3), 1),
    ((-0.3,  0.1,  0.9), 2),
    (( 0.9,  1.2,  0.1), 3),
    (( 0.9,  0.2,  1.1), 4),
    (( 1.3,  0.8,  1.1), 5),

    (( 1.1, -0.1, -0.1), 0),
    (( 0.0,  1.1, -0.1), 1),
    ((-0.1,  0.2,  0.8), 2),
    (( 0.9,  1.1,  0.0), 3),
    (( 0.8, -0.1,  1.0), 4),
    (( 1.2,  0.9,  1.0), 5),
)

print("Training set: " +  str(train_set))

classifier.set_random()
classifier.train_supervised(train_set)

test_set = (
    ((1, 0, 0), 0),
    ((0, 1, 0), 1),
    ((0, 0, 1), 2),
    ((1, 1, 0), 3),
    ((1, 0, 1), 4),
    ((1, 1, 1), 5),

    (( 0.8,  0.1, -0.2), 0),
    (( 0.2,  1.1, -0.3), 1),
    ((-0.3,  0.1,  0.9), 2),
    (( 0.9,  1.2,  0.1), 3),
    (( 0.9,  0.2,  1.1), 4),
    (( 1.3,  0.8,  1.1), 5),

    (( 1.1, -0.1, -0.1), 0),
    (( 0.0,  1.1, -0.1), 1),
    ((-0.1,  0.2,  0.8), 2),
    (( 0.9,  1.1,  0.0), 3),
    (( 0.8, -0.1,  1.0), 4),
    (( 1.2,  0.9,  1.0), 5),

    (( 0.2,  1.0,  1.0), 2),  # a ruse, it belongs to class 5
    (( 0.2,  0.8,  1.0), 2),  # ... but this is class 2 indeed
)

print("Testing set: " + str(test_set))

for vec, expected in test_set:
    c1ass = classifier.classify(vec)
    print(str(vec) + " classifed as " + str(c1ass) + \
        " (" + ("correctly)" if c1ass == expected else "WRONGLY)"))

stats = classifier.test_classifier(test_set)

print("Accuracy: %f" % (stats.accuracy(),))

for c1ass in range(6):
    print("Class %d precision:   %f" % (c1ass, stats.precision(c1ass)))
    print("Class %d recall:      %f" % (c1ass, stats.recall(c1ass)))
    print("Class %d F_1 score:   %f" % (c1ass, stats.F(c1ass)))
    print("Class %d F_0.5 score: %f" % (c1ass, stats.F_beta(0.5, c1ass)))
    print("Class %d F_2 score:   %f" % (c1ass, stats.F_beta(2, c1ass)))

print("F_1   score: %f" % (stats.F(),))
print("F_0.5 score: %f" % (stats.F_beta(0.5),))
print("F_2   score: %f" % (stats.F_beta(2.0),))

if (len(sys.argv) > 1):
    classifier.store(sys.argv[1])
    classifier = lvq.load(sys.argv[1])
    print("Stored and re-loaded to/from " + sys.argv[1])


#
# Clustering
#

data_set = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (1, 1, 1),

    ( 0.8,  0.1, -0.2),
    ( 0.2,  1.1, -0.3),
    (-0.3,  0.1,  0.9),
    ( 0.9,  1.2,  0.1),
    ( 0.9,  0.2,  1.1),
    ( 1.3,  0.8,  1.1),

    ( 1.1, -0.1, -0.1),
    ( 0.0,  1.1, -0.1),
    (-0.1,  0.2,  0.8),
    ( 0.9,  1.1,  0.0),
    ( 0.8, -0.1,  1.0),
    ( 1.2,  0.9,  1.0),
)

print("Data set: " +  str(data_set))

best_ccnt  = 0
least_avge = 999999999
for ccnt in range(1, 10):
    print("Trying with %d clusters..." % (ccnt,))

    clustering = lvq(3, ccnt)

    clustering.set_random()
    clustering.train_unsupervised(data_set)

    for cluster in range(ccnt):
        print("Cluster %d representant: %s" % (cluster, clustering.get(cluster)))

    for vec in data_set:
        cluster = clustering.classify(vec)
        print(str(vec) + " classifed as cluster " + str(cluster))

    stats = clustering.test_clustering(data_set)
    avge  = stats.avg_error()

    print("Avg. error: %f" % (avge,))

    for cluster in range(ccnt):
        print("Cluster %d avg. error: %f" % (cluster, stats.avg_error(cluster)))

    if avge < least_avge:
        least_avge = avge
        best_ccnt  = ccnt

print("Best clustred to %d clusters, avg. error: %f" % (best_ccnt, least_avge))


print("Trying with 6 manually-initialised clusters...")

clustering = lvq(3, 6)

clustering.set(data_set[0], 0)
clustering.set(data_set[1], 1)
clustering.set(data_set[2], 2)
clustering.set(data_set[3], 3)
clustering.set(data_set[4], 4)
clustering.set(data_set[5], 5)
clustering.train_unsupervised(data_set, 20, 9, 1000)

for cluster in range(6):
    print("Cluster %d representant: %s" % (cluster, clustering.get(cluster)))

for vec in data_set:
    cluster = clustering.classify(vec)
    print(str(vec) + " classifed as cluster " + str(cluster))

stats = clustering.test_clustering(data_set)
avge  = stats.avg_error()

print("Avg. error: %f" % (avge,))

for cluster in range(6):
    print("Cluster %d avg. error: %f" % (cluster, stats.avg_error(cluster)))
