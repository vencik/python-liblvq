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

classifier.train(train_set)

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
)

print("Testing set: " + str(test_set))

for vec, _ in test_set:
    print(str(vec) + " classifed as " + str(classifier.classify(vec)))

stats = classifier.test(test_set)

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
