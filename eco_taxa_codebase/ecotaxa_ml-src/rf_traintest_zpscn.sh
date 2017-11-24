#!/usr/bin/env bash
# Run rf_traintest_zp+scn.py on every dataset and grouping.

: "${RESULTS_ROOT:?RESULTS_ROOT needs to be nonempty}"

export PYTHONUNBUFFERED=1

echo Using datasets ${DATASETS:=flowcam uvp5ccelter zoocam zooscan}
echo Using groupings ${GROUPINGS:=group1 group2}
echo Using splits ${SPLITS:=0 1 2}

for DATASET in $DATASETS; do
	export DATASET
	for GROUPING in $GROUPINGS; do
		export GROUPING

		echo $DATASET / $GROUPING

		export SCN_DIR=${RESULTS_ROOT}/SCN_${DATASET}_${GROUPING}_TEST/
	
		if [ ! -d $SCN_DIR ]; then
			echo $RESULTS_DIR is missing. Skipping.
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

		export RESULTS_DIR=${RESULTS_ROOT}/RF_ZP+SCN_${DATASET}_${GROUPING}_TEST/
		mkdir -p $RESULTS_DIR

#			if [ -e $RESULTS_DIR/pca.pickle ]; then
#				echo $RESULTS_DIR/pca.pickle already exists. Skipping.
#				continue
#			fi

		echo RF on ZP+SCN results in $RESULTS_DIR
		
		./rf_traintest_zpscn.py | tee ${RESULTS_DIR}/rf.log
		echo Return status: ${PIPESTATUS[0]} | tee -a ${RESULTS_DIR}/rf.log
	done
done
