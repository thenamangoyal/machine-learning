Lab 4 - Clustering and Dimensionality Reduction
===============================================

Introduction
------------

This project contains 3 parts

1. K-means clustering on MNIST and experimenting on confusion matrix, misclassifications based on number of clusters

2. Principal Component Analysis to reduce the dimensionality of the digit images and effect for reconstruction error based on number of princiapl components chosen.

3. K-means clustering on the data projected onto lower dimensions

* Please refer to [Report.pdf](Report.pdf) for detailed analysis.
* Please refer to [lab.pdf](lab.pdf) for details about the project.
* The code was developed and tested in Matlab 2017a; where in-built K-means clustering was used. If possible, use the same version.

Directory Structure
-------------------
<pre>
---README
---lab.pdf
---Report.pdf
---code
	|
	|---data.txt
	|---label.txt
	|---disptable.m
	|---readdata.m
	|---runclustering.m
	|---runpca.m
	|---Q1.m
	|---Q2.m
	|---Q3.m
---Graphs

</pre>

To Run
------

Change directory to 'code' folder.
In Matlab run scripts for corresponding parts

Q1.m

Q2.m

Q3.m

Graphs generated saved in Graphs


Additional Info
---------------

[ predict] = runclustering( rawX, label ,k)
[ rawprojX, U, transX ] = runpca( rawX , elim)

Naman Goyal
2015CSB1021