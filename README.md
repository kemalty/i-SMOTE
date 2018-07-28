# i-SMOTE

Inverse SMOTE: Minority Oversampling Algorithm with Boundary Extension

i-SMOTE is a modification of the vanilla SMOTE algorithm (https://www.jair.org/index.php/jair/article/view/10302). Instead of filling in space between two minority class instances, i-SMOTE tries to expand the minority class space by adding new instances away from two minority class instances. 

Modification is very simple: just change `Synthetic[newindex][attr] = Sample[i][attr] + gap ∗ dif` to `Synthetic[newindex][attr] = Sample[i][attr] - gap ∗ dif`. 

This modification is related to the open-set classification(https://www.wjscheirer.com/projects/openset-recognition/) as it tries to expand the boundary of a class.

# Installation

Install libraries: `pip install -r requirements.txt`

