#!/usr/bin/env bash
# Run rf_traintest_scn.py on every dataset and grouping.

: "${RESULTS_ROOT:?RESULTS_ROOT needs to be nonempty}"

export PYTHONUNBUFFERED=1

echo Using datasets ${DATASETS:=flowcam uvp5ccelter zoocam zooscan}
echo Using groupings ${GROUPINGS:=group1 group2}

for DATASET in $DATASETS; do
	export DATASET
	for GROUPING in $GROUPINGS; do
		export GROUPING

		echo		
		echo $DATASET / $GROUPING

		export SCN_DIR=${RESULTS_ROOT}/SCN_${DATASET}_${GROUPING}_TEST/

		if [ ! -d $SCN_DIR ]; then
		echo $SCN_DIR is missing. Skipping.
		continue
		fi

		if [ ! -e $SCN_DIR/_train.features ]; then
		echo $SCN_DIR/_train.features is missing. Skipping.
		continue
		fi

		if [ ! -e $SCN_DIR/_test.features ]; then
		echo $SCN_DIR/_test.features is missing. Skipping.
		continue
		fi

		echo SCN Results in $SCN_DIR

		export RESULTS_DIR=${RESULTS_ROOT}/RF_SCN_${DATASET}_${GROUPING}_TEST/
		mkdir -p $RESULTS_DIR

		if [ -e $RESULTS_DIR/rf_model.pickle ]; then
		echo $RESULTS_DIR/rf_model.pickle already exists. Skipping.
		continue
		fi

		echo RF+SCN results in $RESULTS_DIR

		./rf_traintest_scn.py | tee ${RESULTS_DIR}/rf.log
		echo Return status: $? | tee -a ${RESULTS_DIR}/rf.log
	done
done
