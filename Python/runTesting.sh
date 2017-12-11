#!/bin/sh
# Fichier "runTesting"
# Run YOLO on all target repository and launch the python script testingAccuracy afterward

repositorySample="sample_img/"
repositoryDarkflow="darkflow"

cd $repositoryDarkflow
for element in $repositorySample*
   do flow --imgdir $element --model cfg/tiny-yolo-voc-plank.cfg --load 6400 --gpu 1.0
    flow --imgdir $element --model cfg/tiny-yolo-voc-plank.cfg --load 6400 --json --gpu 1.0
done
cd ..

python3 Python/jsonLabel.py $repositoryDarkflow/$repositorySample
python3 Python/testingAccuracy.py $repositoryDarkflow/$repositorySample
