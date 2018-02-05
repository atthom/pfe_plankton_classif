import numpy as np
import os

def create_dir_if_needed(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def similarityToDistance(matrix,epsilone):
    distance_matrix = matrix.copy()
    size = matrix.shape[0]
    for i in range(size):
        for j in range(size):
            if(distance_matrix[i,j] == 0):
                distance_matrix[i,j] = 1/epsilone
            else:
                distance_matrix[i,j] = 1/distance_matrix[i,j]
    return distance_matrix

def count(item):
    item_count = 0
    if type(item) is int:
        item_count = 1
    else:
       item_count = len(item)
    return item_count

def concatanate(item1,item2):
    if ((type(item1) is int) & (type(item2) is int)):
        return [ item1 , item2 ]
    elif ((type(item1) is list) & (type(item2) is list)):
        return item1 + item2
    elif ((type(item1) is int) & (type(item2) is list)):
        return item2 + [item1]
    elif ((type(item1) is list) & (type(item2) is int)):
        return item1 + [item2]

# Search for the the greatest value in the superior triangle of the matrix
def maximumCoefficient(symetric_matrix):
    new_size = symetric_matrix.shape[0]
    maximum = 0
    line = 0
    column = 0
    for i in range(new_size):
        for j in range(i+1,new_size):
            current_value = symetric_matrix[i,j]
            if(current_value > maximum):
                maximum = current_value
                line = i
                column = j
    return maximum,line,column

# Create clustes from the confusion matrix provided according to the parsed arguments threshold and alpha
# return a list containing the cluster created
def confusionGrouping(confusion_matrix,threshold,alpha):
    size = confusion_matrix.shape[0]
    index = list(range(0, size))

    matrix = confusion_matrix.copy()
    print(matrix)
    matrix_transposed = confusion_matrix.copy()
    matrix_transposed = np.transpose(matrix_transposed)

    # We apply power alpha to all elements of the confusion matrix, its transposed matrix and the threshold
    matrix = np.power(matrix, alpha)
    matrix_transposed = np.power(matrix_transposed, alpha)
    threshold = threshold**alpha

    # We create a symetric matrix
    symetric_matrix = 0.5*matrix + 0.5*matrix_transposed
    print(symetric_matrix)

    # All elements inferior to the threshold become 0
    # symetric_matrix[i,j] = 0 imply that the classes i et j are not correlated
    symetric_matrix = np.where(symetric_matrix < threshold, 0, symetric_matrix)
    print(symetric_matrix)
    print(index)

    # Proceed to hierarchical clustering until all symetric_matrix[i,j] > 0 have been taken into account
    maximum,line,column = maximumCoefficient(symetric_matrix)
    while(maximum > 0):

        weight_line = count(index[line])
        weight_column = count(index[column])
        coeff_line = weight_line/(weight_line + weight_column)
        coeff_column = weight_column/(weight_line + weight_column)

        symetric_matrix[line,:] = symetric_matrix[line,:]*coeff_line + symetric_matrix[column,:]*coeff_column
        symetric_matrix[:,line] = symetric_matrix[:,line]*coeff_line + symetric_matrix[:,column]*coeff_column
        symetric_matrix = np.delete(symetric_matrix,column, 0)
        symetric_matrix = np.delete(symetric_matrix,column, 1)

        index[line] = concatanate(index[line],index[column])
        del index[column]

        print(symetric_matrix)
        print(index)

        maximum,line,column = maximumCoefficient(symetric_matrix)
        symetric_matrix = np.where(symetric_matrix < threshold, 0, symetric_matrix)

    return index

# Create a new hierarchical repository from an old one and copy all its files.
def newRepository(index,species,target_path,new_path):
    for i in range(len(index)):
        if(type(index[i]) is int):
            dirname = species[index[i]]
            path_destination_files = new_path + "/" + dirname
            path_files = target_path + "/" + dirname
            create_dir_if_needed(path_destination_files)
            for fic in os.listdir(path_files):
                shutil.copy2((path_files + "/" + fic),path_destination_files)
        else:
            super_dirname = ""
            for j in range(len(index[i])):
                super_dirname = super_dirname + species[index[i][j]]
            for j in range(len(index[i])):
                dirname = species[index[i][j]]
                path_destination_files = new_path + "/" + super_dirname + "/" + dirname
                path_files = target_path + "/" + dirname
                create_dir_if_needed(path_destination_files)
                for fic in os.listdir(path_files):
                    shutil.copy2((path_files + "/" + fic),path_destination_files)


def main():
    species = ["requin","baleine","poisson","crevette","noixdecoco","pierre","anguille"]
    confusion_matrix = np.array([[0.7, 0.1, 0.1,0.025,0.025,0.025,0.025],
     [0.09, 0.7, 0.11,0.025,0.025,0.025,0.025],
     [0.15, 0.05, 0.7,0.025,0.025,0.025,0.025],
     [0.05, 0.05, 0.05,0.7,0.05,0.05,0.05],
     [0.05, 0.05, 0.05,0.05,0.4,0.35,0.05],
     [0.05, 0.05, 0.05,0.05,0.4,0.35,0.05],
     [0.05, 0.05, 0.05,0.05,0.05,0.05,0.7],
     ])
    threshold = 0.07
    alpha = 0.5
    index = confusionGrouping(confusion_matrix,threshold,alpha)
    target_path = "../../test"
    new_path =  "../../newtest"
    newRepository(index,species,target_path,new_path)


main()
