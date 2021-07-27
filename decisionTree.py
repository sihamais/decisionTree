# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sn
import numpy as np
from sklearn.cluster import KMeans

"""
## II. Données
"""
dataFile = open("data.csv", "r")
data = pd.read_csv(dataFile, sep="\t", names=["X1", "X2","GroundTruth"])

print("\nIII. Évaluation")
"""
## III. Évaluation

2. Calculez puis comparez l’exactitude et l’exactitude pondérée de cette matrice. Commentez.
"""
TN = 1000; FP = 2; FN = 30; TP = 5;

def getAccuracy(TN,FP,FN,TP):
  accuracy = (TN + TP) / (TN + FP + FN + TP)
  print("Accuracy :",accuracy)

def getWeightedAccuracy(TN,FP,FN,TP):
  weighted_accuracy = ((TN / (TN + FP)) + (TP /(FN + TP))) / 2
  print("Weighted Accuracy :", weighted_accuracy)
  return weighted_accuracy

def getPrecision(FP,TP):
  precision = TP / (TP + FP)
  print("Precision :",precision)
  return precision

def getRecall(FN,TP):
  recall = TP / (FN + TP)
  print("Recall :",recall)
  return recall

getAccuracy(TN,FP,FN,TP)
getWeightedAccuracy(TN,FP,FN,TP)

"""
## IV. Algorithmes

### 4.1 Arbre réduit à 1 feuille (classe DecisionLeaf )

1. Définissez une structure de données adaptée pour représenter une telle
feuille.
"""
print("\n4.1 Arbre réduit à 1 feuille (classe DecisionLeaf )")
class DecisionLeaf :

  def __init__(self, D, attribIdx):
    self.D = D
    if len(attribIdx)>1:
      # Find largest standard deviation if more than one feature
      feature, a, b = getSplitParameters(self.D, attribIdx)
      self.attribIdx = feature
      self.a = a;
      self.b = b
    else :
      # Get cluster values if only one feature
      self.attribIdx = attribIdx[0]
      self.splitParams()


  def splitParams(self):
     # Get initial centroids
    minInit = min(self.D.iloc[:,self.attribIdx])
    maxInit = max(self.D.iloc[:,self.attribIdx])

    # Get clusters centers
    initialCluster = np.array([minInit,maxInit]).reshape(2,1)
    kmeans = KMeans(n_clusters=2, init=initialCluster, n_init=1).fit(self.D.iloc[:,self.attribIdx].values.reshape(-1,1))
    a = kmeans.cluster_centers_[0][0]
    b = kmeans.cluster_centers_[1][0]

    self.a = a
    self.b = b

  # Predict class for a row
  def infer(self,row):
    if row.iloc[self.attribIdx] <= self.a :
      return 1
    elif row.iloc[self.attribIdx] > self.b :
      return 1
    else :
      return 0 
  
  def traversal(self):
    tab = []
    # Get prediction for each row
    for i in range(self.D.shape[0]):
      tab.append(self.infer(self.D.iloc[i]))

    # Replace GroundTruth values with predicted values
    data_copy = self.D.copy()
    data_copy.drop(['GroundTruth'], axis=1)
    data_copy['GroundTruth'] = tab
    return data_copy



def getSplitParameters(D,attribIdx):
    maxStd = 0

    # Get current attribute
    for idx in attribIdx:
      stdcurr = np.std(D.iloc[:, idx])
      if stdcurr > maxStd :
        maxStd = stdcurr
        featureLabel = D.columns.values[idx]

    # Get initial centroids
    minInit = min(D[featureLabel])
    maxInit = max(D[featureLabel])

    # Get clusters centers
    initialCluster = np.array([minInit,maxInit]).reshape(2,1)
    kmeans = KMeans(n_clusters=2, init=initialCluster, n_init=1).fit(D[featureLabel].values.reshape(-1,1))
    a = kmeans.cluster_centers_[0][0]
    b = kmeans.cluster_centers_[1][0]

    return D.columns.get_loc(featureLabel), a, b

leaf = DecisionLeaf(data,[0])
predictedData = leaf.traversal()

"""
3. Évaluez le modèle appris
"""
print("\n3. Évaluez le modèle appris")
def evaluate(D, predictionD):
  TN = 0; TP = 0; FN = 0; FP = 0;

  # Create table for comparison between ground truth and predicted class
  comparisonTable = D['GroundTruth'].compare(predictionD['GroundTruth'], keep_shape=True, keep_equal=True)
  size = len(comparisonTable)

  # Fill confusion matrix
  for i in range(size):
    if (comparisonTable['self'][i]==0) & (comparisonTable['other'][i]==0) :
      TN+=1
    elif (comparisonTable['self'][i]==0) & (comparisonTable['other'][i]==1) :
      FP+=1
    elif (comparisonTable['self'][i]==1) & (comparisonTable['other'][i]==1) :
      TP+=1
    else :
      FN+=1

  # Get evaluation
  WA = getWeightedAccuracy(TN,FP,FN,TP)
  PR = getPrecision(FP,TP)
  RC = getRecall(FN,TP)

  ax = sn.scatterplot(x='X1', y='X2', hue='GroundTruth', data=predictionD, palette='Set2')

  return WA, PR, RC

evaluate(data, predictedData);

print("\n4.2 Arbre superficiel")
"""### 4.2 Arbre superficiel

1. Précisez les structures de données utilisées
"""
class Node:
  def __init__(self, currentAttrib, a, b, L, M, R):
    self.currentAttrib = currentAttrib
    self.a = a
    self.b = b
    self.L = L
    self.M = M
    self.R = R

  def infer(self, row) :
    if row.iloc[self.currentAttrib] <= self.a :
      return self.L.infer(row)
    elif row.iloc[self.currentAttrib] > self.b :
      return self.R.infer(row)
    else :
      return self.M.infer(row)

class DirectDecision:
  def __init__(self, outlier):
    self.outlier = outlier

  def infer(self, row):
    # Reverse values (inliers : 0, outliers: 1)
    return int(not self.outlier)

"""2. Implémentez la méthode proposée"""
def buildDecisionTree(D, central, attribIdx):
  if len(D) >= 4 :
    if len(attribIdx) >= 2 :

      currentAttrib, a, b = getSplitParameters(D, attribIdx)
      attribIdx.remove(currentAttrib)

      Dl = D[D.iloc[:,currentAttrib] <= a]
      Dm = D[(D.iloc[:,currentAttrib] > a) & (D.iloc[:,currentAttrib] <= b)]
      Dr = D[D.iloc[:,currentAttrib] > b]

      L = buildDecisionTree(Dl, False, attribIdx)
      M = buildDecisionTree(Dm, True, attribIdx)
      R = buildDecisionTree(Dr, False, attribIdx)

      return Node(currentAttrib, a, b, L, M, R)

    else :
      return DecisionLeaf(D, attribIdx)

  else :
    if central == True :
      return DirectDecision(outlier = False)

    else :
      return DirectDecision(outlier = True)

def treeTraversal(decisionTree,D):
  tab = []
  for i in range(D.shape[0]):
    tab.append(decisionTree.infer(D.iloc[i]))

  data_copy = D.copy()
  data_copy.drop(['GroundTruth'], axis=1)
  data_copy['GroundTruth'] = tab
  return data_copy

"""3. Évaluez votre modèle et discutez les résultats."""
print("\n3. Évaluez votre modèle et discutez les résultats.")
decisionTree = buildDecisionTree(data, True, [0,1])
predictedData = treeTraversal(decisionTree,data)

evaluate(data, predictedData);

print("\n4.3 Arbre généralisé")
"""### 4.3 Arbre généralisé

1. Adapter l’algorithme 1 pour pouvoir réutiliser le même attribut plusieurs fois (et ajoutez le nouveau critère d’arrêt).
"""

def buildHeightDecisionTree(D, central, attribIdx, hauteurMax):
  if len(D) >= 4 :
    if hauteurMax > 1 :

      currentAttrib, a, b = getSplitParameters(D, attribIdx)

      Dl = D[D.iloc[:,currentAttrib] <= a]
      Dm = D[(D.iloc[:,currentAttrib] > a) & (D.iloc[:,currentAttrib] <= b)]
      Dr = D[D.iloc[:,currentAttrib] > b]

      L = buildHeightDecisionTree(Dl, False, attribIdx, hauteurMax-1)
      M = buildHeightDecisionTree(Dm, True, attribIdx, hauteurMax-1)
      R = buildHeightDecisionTree(Dr, False, attribIdx, hauteurMax-1)

      return Node(currentAttrib, a, b, L, M, R)

    else :
      return DecisionLeaf(D, attribIdx)

  else :
    if central == True :
      return DirectDecision(outlier = False)

    else :
      return DirectDecision(outlier = True)


print("\n2. Évaluez le modèle avec des hauteurs maximales de 1 à 4 inclus")
"""2. Évaluez le modèle avec des hauteurs maximales de 1 à 4 inclus et réalisez un tableau reprenant l’ensemble des métriques demandées : exactitude pondérée, précision et rappel. Vérifiez que le cas h = 1 correspond à une feuille de décision (vous devriez obtenir exactement le même modèle qu’à la section n 4.1 : mêmes seuils, même indice d’attribut mêmes métriques, etc.)."""

WAlist = []
PRlist = []
RClist = []

decisionTree = buildHeightDecisionTree(data, True, [0,1], 1)
predictedData = treeTraversal(decisionTree,data)
WA, PR, RC = evaluate(data, predictedData);
WAlist.append(WA)
PRlist.append(PR)
RClist.append(RC)

decisionTree = buildHeightDecisionTree(data, True, [0,1], 2)
predictedData = treeTraversal(decisionTree,data)
WA, PR, RC = evaluate(data, predictedData);
WAlist.append(WA)
PRlist.append(PR)
RClist.append(RC)

decisionTree = buildHeightDecisionTree(data, True, [0,1], 3)
predictedData = treeTraversal(decisionTree,data)
WA, PR, RC = evaluate(data, predictedData);
WAlist.append(WA)
PRlist.append(PR)
RClist.append(RC)

decisionTree = buildHeightDecisionTree(data, True, [0,1], 4)
predictedData = treeTraversal(decisionTree,data)
WA, PR, RC = evaluate(data, predictedData);
WAlist.append(WA)
PRlist.append(PR)
RClist.append(RC)

stats = pd.DataFrame({'Weighted Accuracy': WAlist, 'Precision': PRlist, 'Recall': RClist}, index=['Height = 1','Height = 2','Height = 3', 'Height = 4'])
print(stats)