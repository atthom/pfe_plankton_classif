# imports
import os
import sys
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Taking argument into account to find out the repository to be processed
if len(sys.argv) >= 1:
    repository = sys.argv[1]
else:
    print("Python need the path of the repository containing the images to be tested")
    sys.exit(0)

# Merge the information of right and left parsed sequence
def readJson(repository):
    dir_names = [_ for _ in os.listdir(repository)]
    nbrOfClasses = len(dir_names)
    classificationMatrix = np.zeros((nbrOfClasses,nbrOfClasses+1))

    for index_true_label in range(nbrOfClasses):
        dirname = dir_names[index_true_label]
        Dir = repository + dirname
        liste = sorted(glob.glob(Dir + "/[0-9]*.json"))
        liste = [l.split('/')[-1] for l in liste]
        print(liste)

        for l in liste:

            # Extract informations
            json_data = open(Dir+"/"+l)
            data = json.load(json_data)

            label = []
            topleftCorner = []
            bottomrightCorner = []

            for i in range(len(data)):
                label.append(data[i]["label"])
                topleftCorner.append([data[i]["topleft"]["x"],data[i]["topleft"]["y"]])
                bottomrightCorner.append([data[i]["bottomright"]["x"],data[i]["bottomright"]["y"]])
                json_data.close()

            json_data_predicted = open(Dir+"/out/"+l)
            data_predicted = json.load(json_data_predicted)

            label_predicted = []
            confidence_predicted = []
            topleftCorner_predicted = []
            bottomrightCorner_predicted = []

            for i in range(len(data_predicted)):
                label_predicted.append(data_predicted[i]["label"])
                confidence_predicted.append(data_predicted[i]["confidence"])
                topleftCorner_predicted.append([data_predicted[i]["topleft"]["x"],data_predicted[i]["topleft"]["y"]])
                bottomrightCorner_predicted.append([data_predicted[i]["bottomright"]["x"],data_predicted[i]["bottomright"]["y"]])
                json_data_predicted.close()

            # find the best match
            for i in range(len(label)):
                match = []
                matchingscore = []
                area = (bottomrightCorner[i][0]-topleftCorner[i][0])*(bottomrightCorner[i][1]-topleftCorner[i][1])
                for j in range(len(label_predicted)):
                    commonWidth = min(bottomrightCorner_predicted[j][0],bottomrightCorner[i][0]) - max(topleftCorner_predicted[j][0],topleftCorner[i][0]);
                    commonHeight = min(bottomrightCorner_predicted[j][1],bottomrightCorner[i][1]) - max(topleftCorner_predicted[j][1],topleftCorner[i][1]);
                    common_area = max(0,commonWidth)*max(0,commonHeight)
                    if(common_area >= 0.5*area):
                        match.append(j)
                        matchingscore.append(common_area)

                # no match were found between the predicted and original label
                if(len(match)==0):
                    classificationMatrix[index_true_label][nbrOfClasses] = classificationMatrix[index_true_label][nbrOfClasses] + 1
                # one match was found
                if(len(match)==1):
                    indice = match[0]
                    label = label_predicted[indice]
                    index_predicted_label = dir_names.index(label)
                    classificationMatrix[index_true_label][index_predicted_label] = classificationMatrix[index_true_label][index_predicted_label] + 1

                # more than one match was found, we will keep the one with the best matching score
                if(len(match)>1):
                    indice = 0
                    score = matchingscore[0]
                    for k in range(len(match)):
                        newScore = matchingscore[k]
                        if(newScore > score):
                            score = newScore
                            indice = k
                            indice = match[indice]
                    label = label_predicted[indice]
                    index_predicted_label = dir_names.index(label)
                    classificationMatrix[index_true_label][index_predicted_label] = classificationMatrix[index_true_label][index_predicted_label] + 1

    return classificationMatrix,dir_names

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    classificationMatrix,classes = readJson(repository);
    print(classificationMatrix)
    size = classificationMatrix.shape
    confusionMatrix = np.delete(classificationMatrix, (size[1]-1), axis=1)
    accuracy = np.trace(confusionMatrix)/np.sum(confusionMatrix)
    print(confusionMatrix)
    print(classes)
    print(accuracy)

    plt.figure()
    plot_confusion_matrix(confusionMatrix, classes, normalize=True,
                      title='Normalized confusion matrix')
    plt.show()


main()
print("\n\ndone.")
