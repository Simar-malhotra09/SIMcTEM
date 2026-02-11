## Synopsis: 

Biggest bottleneck in instance level segmentation of CTEM micrographs is the process of manual labelling of training data. Particles are often overlapped, which requires domain expertise making the task more expensive.
The idea here is to make a CNN approximate the shape of a certain kind of sample (like Molybdenum) through regular shapes (like a circle) and adding psuedo randomness (intensity or structural defects) and by adding noise to the background.

Poor man's Physics informed modelling, so to say. 
