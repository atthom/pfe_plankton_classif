#!/usr/bin/env bash
# Run rf_traintest_zooprocess.py on every dataset and grouping.

: "${DATASET_ROOT:?DATASET_ROOT needs to be nonempty}"
: "${RESULTS_ROOT:?RESULTS_ROOT needs to be nonempty}"

export PYTHONUNBUFFERED=1

echo Using datasets ${DATASETS:=flowcam uvp5ccelter zoocam zooscan}
echo Using groupings ${GROUPINGS:=group1 group2}

for DATASET in $DATASETS; do
	export DATASET
	echo Dataset $DATASET
	for GROUPING in $GROUPINGS; do
		export GROUPING
		echo Grouping $GROUPING

		export RESULTS_DIR=${RESULTS_ROOT}/RF_${DATASET}_${GROUPING}_TEST/
		mkdir -p $RESULTS_DIR

		echo Results are written to $RESULTS_DIR

		./rf_traintest_zooprocess.py | tee ${RESULTS_DIR}/random_forest.log
		echo Return status: $? | tee -a ${RESULTS_DIR}/random_forest.log
	done
done
