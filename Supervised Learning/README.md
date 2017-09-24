Lab1 - Decision Tree
====================

Introduction
------------

Experimenting with decision tree based inductive classifier using early stopping, noise, post pruning, random forests.

* Please refer to [Report.pdf](Report.pdf) for detailed analysis.
* Please refer to [lab.pdf](lab.pdf) for about the project.

Directory Structure
-------------------
<pre>
---README
---lab.pdf
---Report.pdf
---code
	|
	|--decision.cpp
	|--imdbEr.txt
	|--imdb.vocab
	|--makefile
	|--script.sh
	|--script_skip1.sh
	|--selected-features-indices.txt
	|
	|--train
	|	|
	|	|--labeledBow.feat
	|
	|
	|--test
		|
		|--labeledBow.feat
</pre>

Executing
---------

Make the code using
$make
or
$g++ decision.cpp -o decision

Exceute using
./decision filename expno polarity
-- filename refers to the selected indices file
-- refers to expno from 1-5
-- polarity is optional. 0 (default) = use most frequent words; 1 = use highest and lowest polarity words

You can use script to run all functionality using
$./script.sh

To use previously complied selected-features-indices.txt and skip expno 1 use
$./script_skip1.sh

Note: It script fails to exceute use
$chmod 755 script.sh

Feel free to email at 2015csb1021@iitrpr.ac.in for any clarification.

Developed by
Naman Goyal
2015CSB1021
