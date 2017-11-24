# Classification of zooplankton images

## Preparation of input data

The input data are greyscale images and tables of features measured on these images; both are stored under `$DATASET_ROOT` (outside this repo). These span several datasets (flowcam, uvp5ccelter, zoocam, zooscan). One dataset consists of the image data in `$DATASET_ROOT/<dataset>/<category>/<objid>.jpg` and the features table `$DATASET_ROOT/<dataset>_data.csv`.  

As these scripts construct the training and testing data for all experiments, they MUST NOT BE RUN after the experiments have started (NB: the data is split deterministically, so if the input data does not change, the splits will not change either).

The steps to create the training data are documented in `setup_training_data.sh`.

`classes_from_taxa` create a file with a list of classes in order, which are then used by SparseConvNet or Random Forest as indexes for classes (by reading the respective column in the taxa mapping file)

`dataset_add_***` add information to features tables

`split_dataset` split a dataset between train and test, stratified by original class. Takes `$DATASET_ROOT/<dataset>_data.csv` and creates `$DATASET_ROOT/<dataset>_data_train.csv` and `$DATASET_ROOT/<dataset>_data_test.csv`.

`***_collection_***` split each training dataset into + 3-folds. Creates indexes for test and train (with folds).

`extract_columns` read features files. Used to store the assignment of objects to training and test sets and the validation splits to be able to recreate these sets without storing the whole data. The results lie in [ecotaxa-data](https://bitbucket.org/moi90/ecotaxa-data).

`util/features.py` load SparseConvNet features

`util/zooscan.py` load ZooProcess features

## Experiments

`call_scn` call SparseConvNet to train a network, predict identifications and extract objects features

`rf_***` train a RandomForest model on all datasets

`***_scn` train on SparseConvNet features

`***_zp` train on ZooProcess features

`***_zp+scn` train on both

`***_trainval_***` train on 2/3 of training set and validate on the last one (use all combinations = cross validate); use a grid of RF parameters to find the best combination

`***_traintest_***`use the best combination of parameters, train on the full training set, predict the test set

`util/features` perform PCA on SCN features

`find_best_scn_split.py` For each dataset and grouping, identify the best performing validation split that can be later evaluated on the testing set.

### Environment variables

A shell file sets the environment variables and then calls the appropriate python file, for each experiment

`DATASET_ROOT` Directory where all the data is stored.

`RESULTS_ROOT` Directory where all the results are stored.

`SCN_DIR` Set by `rf_*scn*`. Directory of a single SCN experiment, where feature files can be found.

`RESULTS_DIR` Directory of a single experiment, where features and weights (in case of SCN) and logs, confusion matrices, ... are stored.

## Diagnostics

`parse_wp2_log`, `wp2_show_progress` extract and display training progress from SCN

`assemble_results` read confusion matrices and logs and compute global accuracy, for all datasets. The results include the mean values of the validation splits and the evaluation on the held-out testing set.

`plot_***` show results.

## Exported Models

### SCN models

- classes.txt (Classes in order)
- _epoch-99.cnn (Weights for the CNN)
- feature_pca.jbl (Joblib dump of the PCA model trained on the SCN features of the training images, n_compontents=50)
- meta.json:
  - epoch: Epoch of the weights file
  - type: "SparseConvNet"
  - name: Unique name for this model


### RF models

- classes.txt (Classes in order)

- random_forest.jbl (Joblib dump of the RF model trained on the selected ZooProcess fields and the PCA-compressed SCN features)

- meta.json:

  - zooprocess_fields: Ordered list of ZooProcess fields / features, that were used in RF training
  - type: "RandomForest-ZooProcess+SparseConvNet"
  - n_objects: Number of objects in the training set
  - zooprocess_medians: List of medians for each ZooProcess field (used for imputing missing values)
  - name: Uninque name
  - scn_model: SCN model that generated the features

The RF models was trained on the concatenation of all `zooprocess_fields`  with the PCA-compressed SCN features so that `X.shape = (n_objects, len(zooprocess_fields) + X_scn.shape[1])`.