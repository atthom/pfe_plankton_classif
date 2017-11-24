#!/usr/bin/env bash
# Set up the training data

echo "Read and edit this script before executing."
echo "Don't run this script more than once!"
exit 1

# Directory that contains the images and the data exports from Ecotaxa and the taxonomic mappings:
# - <dataset>_data.csv
# - <dataset>/<category>/<objid>.jpg
# - <dataset>_taxa.csv
# It will be filled with additional data
export DATASET_ROOT=/data-ssd/mschroeder/Ecotaxa/Datasets/ # on ob, or
#export DATASET_ROOT=/home/mschroeder/Ecotaxa/Datasets/ # on niko

# Directory where the results of the experiments will be stored
export RESULTS_ROOT=/data1/mschroeder/Ecotaxa/Results/ # on ob, or
# export RESULTS_ROOT=/home/mschroeder/Ecotaxa/Results/ # on niko
mkdir -p ${RESULTS_ROOT}

# CD into ${RESULTS_ROOT}
cd ${RESULTS_ROOT}

# Directory with all the scripts
SCRIPT_DIR=/home/mschroeder/Ecotaxa/src

# Disable output buffering to see the output in spite of `tee`
export PYTHONUNBUFFERED=1

for DATASET in flowcam uvp5ccelter zoocam zooscan; do
	export DATASET

	echo
	echo $DATASET

	# Augment a ZooProcess dataset with columns `name_*` containing the object label name according to group1 and group1
	${SCRIPT_DIR}/dataset_add_groupings.py ${DATASET}_data.csv ${DATASET}_data_groupings.csv ${DATASET}_taxa.csv --columns group1,group2 | tee ${DATASET}_groupings.log

	# Split dataset into training and testing data
	split_dataset.py ${DATASET}_data_groupings.csv ${DATASET}_data_train.csv ${DATASET}_data_test.csv | tee ${DATASET}_split.log

	for GROUPING in group1 group2; do
		export GROUPING

		echo		
		echo $DATASET / $GROUPING

		# Assemble the class list from the taxonomic mapping
		classes_from_taxa.py ${DATASET}_taxa.csv ${GROUPING} ${DATASET}_classes_${GROUPING}.txt | tee ${DATASET}_classes_${GROUPING}.log

		# Assemble image collection indices for SCN training and validation
		create_collection_index.py --effective-name name_${GROUPING} --cross-validation 3 ${DATASET}_data_train.csv ${DATASET}/ ${DATASET}_${GROUPING}_collection | tee ${DATASET}_${GROUPING}_collection.log

		# Assemble image collection indices for SCN testing
		create_collection_index.py --effective-name name_${GROUPING} ${DATASET}_data_test.csv ${DATASET}/ ${DATASET}_${GROUPING}_collection_test | tee ${DATASET}_${GROUPING}_collection_test.log

		# Create image collection directories for SCN training and testing
		create_collection_dir.py ${DATASET}_${GROUPING}_collection*.csv | tee ${DATASET}_collection_dir.log
	done
done


