Lab3 - Neural Network and Self Driving Cars
===========================================

Introduction
------------

This project contains two parts

1.Experimenting with multi-layer perceptron. A simple 2 dimensional
3 class classification problem to visualize decision boundaries learned by a MLP. The second part of the lab
takes time to train

2. Neural network to predict the steering angle the road image for a self-driving car application that is inspired by Udacity’s Behavior. Performed the following modifications to the networks to try better models with the constraint that these were implemented from scratch. (some of them are easy to implement, while some took a significant amount of time)
Implement convolution layers
Implement a different activation function
Implement Max Pooling layers
Implement and try a different optimizer (such as Adam or RMS Prop)

Note
----
Code was tested on Matlab 2017a.

Final result in Prediction.txt

* Please refer to [Report.pdf](Report.pdf) for detailed analysis.
* Please refer to [lab.pdf](lab.pdf) for details about the project.
* The code was developed and tested in Matlab 2017a. If possible, use the same version.

Directory Structure
-------------------
<pre>
---README
---lab.pdf
---Report.pdf
---code
	|--steering (train data)
	|--test (test data)
	|--l3a.m
	|--generate_report.m
	|--q2_adam.m
	|--q2_rms.m
	|--q2_inverteddropout.m
</pre>

Part 1 (MLP)
----------------

Run l3a.m

Part 2 (Self Drving Car - Generate Report)
------------------------------------------

Run generate_report.m


Part 2 (Modifications for Competition)
--------------


Run q2_adam, q2_rms, q2_inverteddropout

Best results with q2_adam

Developed by
Naman Goyal

