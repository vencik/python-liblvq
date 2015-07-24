#!/usr/bin/env python

import sys;
sys.path.append("/home/vencik/tmp/prefix/lib/python2.7/site-packages");

from liblvq import lvq;


##
#  \brief  Print (cluster, weight) tuples list
#
#  \param  cws  (cluster, weight) tuples list
#
def printCWlist(cws):
    for cw in cws:
        print "Cluster " + str(cw[0]) + ": " + str(cw[1]);

weight = (0.75, 0.20, 0.05);

#printCWlist(lvq.best(weight, 3));
#printCWlist(lvq.best(weight, 2));
#printCWlist(lvq.best(weight));

classifier = lvq(3, 6);
#classifier = lvq();

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
);

classifier.train(train_set, 5, 3, 1000);

print "Training set: " +  str(train_set);
print "Learn rate: " + str(classifier.learn_rate(train_set));

test_set = (
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
);

for vec in test_set:
    print str(vec) + " classifed as " + str(classifier.classify(vec));

#print "Testing set: " + str(test_set);
#print "Weights: " + str(weight);
