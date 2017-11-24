#!/usr/bin/env bash

# Use extract_columns.py to save the data that is required to restore
# the training, validation and testing data
#
# ${DATASET_ROOT}/${DATASET}_index_{train,test}.csv is formed from ${DATASET}_data_{train,test}.csv
# The train file contains obj and split, the test file only contains objid.


: "${DATASET_ROOT:?DATASET_ROOT needs to be nonempty}"

export PYTHONUNBUFFERED=1

for DATASET in flowcam uvp5ccelter zoocam zooscan; do
	echo Dataset $DATASET
	./extract_columns.py ${DATASET_ROOT}/${DATASET}_data_train.csv ${DATASET_ROOT}/${DATASET}_index_train.csv objid split
	./extract_columns.py ${DATASET_ROOT}/${DATASET}_data_test.csv ${DATASET_ROOT}/${DATASET}_index_test.csv objid
done
