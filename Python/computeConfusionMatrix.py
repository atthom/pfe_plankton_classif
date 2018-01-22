# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:53:24 2018

@author: Cédric
"""

import numpy as np

def confusionMatrix(confMat, yTrue, yPred, nbClasses):
    
    for i in range (0,yPred.shape[0]):
        maxIndex = np.argmax(yPred[i,:])
        yPred[i,maxIndex] = 1
    trueClasses = np.argwhere(yTrue==1)[:,1]
    predClasses = np.argwhere(yPred==1)[:,1]
    for k in range (0,trueClasses.size):
        i = trueClasses[k]
        j = predClasses[k]
        confMat[i][j] = confMat[i][j] + 1
    
    return confMat
    
# Excample 
def main():
    
    nbClasses = 3
    
    confu = np.zeros((nbClasses,nbClasses),dtype=float)
    
    true = np.array([[1,0,0],[0,1,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[1,0,0],[0,0,1],[0,0,1]])

    pred = np.array([[0.667,0.333,0],[0,1.0,0],[0,0.9,0.1],[0.47,0.33,0.2],[0,1.0,0],[0.33,0.33,0.33],[0,0,1],[1,0,0],[1,0,0],[0,0,1]])

    confu = confusionMatrix(confu,true,pred,nbClasses)
    
    #Une fois que l'on a parcouru toute les images il faut ensuite normaliser la matrice de confusion
    
    for i in range(0,nbClasses):
        confu[i,:] = confu[i,:] / sum(confu[i,:])
        
        
    

