import os
from LayerClassifier import LayerClassifier

super_path = "E:\\Polytech_Projects\\pfe_plankton_classif\\LOOV\\super_classif"
super_path = "E:\\Polytech_Projects\\pfe_plankton_classif\\Dataset\\DATASET\\level0_new_hierarchique"

lc = LayerClassifier(super_path)
lc.create_achitecture(500, 5)

# lc.load_achitecture()
