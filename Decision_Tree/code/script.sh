#!/usr/bin/env bash
make
for i in {1..5}
do
	echo "Running expno ${i}"
	./decision selected-features-indices.txt ${i}
done
