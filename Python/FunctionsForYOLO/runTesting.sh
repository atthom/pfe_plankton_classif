#!/bin/sh
# Fichier "runTesting"
# Run YOLO on all target repository and launch the python script testingAccuracy afterward

repositorySample="sample_img/"
repositoryDarkflow="darkflow"

cd $repositoryDarkflow
for element in $repositorySample*
   do python flow --imgdir $element --model cfg/tiny-yolo-voc-plank2.cfg --load 110400 --gpu 0.9
    python flow --imgdir $element --model cfg/tiny-yolo-voc-plank2.cfg --load 110400 --json --gpu 0.9
done
cd ..

python Python/jsonLabel.py $repositoryDarkflow/$repositorySample
python Python/testingAccuracy.py $repositoryDarkflow/$repositorySample
