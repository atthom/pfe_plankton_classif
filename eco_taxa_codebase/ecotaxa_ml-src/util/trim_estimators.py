# -*- coding: utf-8 -*-
"""
Trim estimators. This enables to train a large estimator once and later evaluate
trimmed versions.
"""

from copy import deepcopy

def trim_pca(pca, keep_n_components):
    """
    Trim a PCA model to the desired number of components.
    """
    pca = deepcopy(pca)
    
    # TODO: Anything more?
    pca.explained_variance_ = pca.explained_variance_[:keep_n_components].copy()
    pca.components_ = pca.components_[:keep_n_components].copy()
    
    return pca

def trim_rf(rf, keep_n_estimators):
    """
    Trim a Random Forest model to the desired number of estimators.
    
    DEPRECATED: Rather use sklearn.ensemble.RandomForestClassifier with warm_start=True.
    """
    rf = deepcopy(rf)
    
    # TODO: Anything more?
    rf.n_estimators = keep_n_estimators
    rf.estimators_ = rf.estimators_[:keep_n_estimators]
    
    return rf