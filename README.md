<center>
<h1><b>Détection d’outliers par arbres de décisions</b></h1>
</center>

## Données
Le set de données comporte 250 ***inliers*** et 80 ***outliers***, soit 330 exemples.

<div style="display: grid;grid-template-columns: 1fr 2fr; grid-gap: 40px;margin:auto;">
<div style="margin-top:auto; margin-bottom:auto;margin-right:30px;">
<ul>
<li>Accuracy : <br>0.9691417550626809</li>
<li>Weighted Accuracy :<br>0.5704305674365554</li>
</ul>
</div>
<img src="Data.png">
</div>

On peut constater que les données catégorisées comme ***inliers*** (*GroundTruth* = 0) sont regroupés au centre du graphe. Tandis que les ***outliers*** (*GroundTruth* = 1) sont dispérsées autour des Inliers.


## Évaluation

Ce modèle arrive à prédire correctement un grand nombre d'***inliers***. Cependant il n'arrive à prédire correctement que 5 ***outliers*** avec 30 mauvaises prédictions. Donc ce modèle n'est pas bon.

On obtient le pourcentage de l'exactitude ( les bonnes préditions ) d'environ 96.92%.
L'exactitude est très haute car le modèle prédit correctement presque tout les ***inliers*** qui consituent la quasi totalité des exemples. 
Par contre, on a obtenu le pourcentage d'environ 57% pour l'exactitude pondérée ( bonne prédictions pour chaque classe ), qui considère l'erreur sur les ***outliers*** avec un poids égal à celui des erreurs sur les ***inliers***, elle donne donc une mesure plus intéressante dans notre cas.

L'exactitude donne un score aussi bon car elle se base sur la majorité des données qui sont des ***inliers***, elle ne considère donc que très peu les ***outliers*** dans son calcul. Le modèle ne se trompe donc que très peu car presque tout les exemples sont des ***inliers***.

Elle n'est pas pertinente dans notre cas car le dataset n'est pas balancé. Le nombre de données labelisés ***inliers*** est largement supérieur au nombre de données labelisé ***outliers***. 

## Algorithmes

### Arbre réduit à 1 feuille (classe DecisionLeaf)
<div style="display: grid;grid-template-columns: 1fr 2fr; grid-gap: 40px;margin:auto;">
<div style="margin-top:auto; margin-bottom:auto;margin-right:30px;">
<ul>
<li>Weighted Accuracy :<br> 0.758</li>
<li>Precision :<br> 0.5490196078431373</li>
<li>Recall :<br> 0.7</li>
</ul>
</div>
<img src="4.1_data.png">
</div>

### Arbre superficiel
<div style="display: grid; grid-gap:40px;margin:auto;grid-template-columns: 1fr 2fr;">
<div style="margin-top:auto; margin-bottom:auto;margin-right:30px">
<ul>
<li>Weighted Accuracy :<br> 0.79425</li>
<li>Precision :<br> 0.6477272727272727</li>
<li>Recall :<br> 0.7125</li>
</ul>
<p></p>
</div>
<img src="4.2_data.png">
</div>
L'évaluation obtenue du modèle est meilleure avec l'algorithme <b>ID3</b> puisqu'il permet de mieux englober les données en limitant l'utilisation d'un attribut à une seule fois. Les attributs sont donc tous pris en compte d'où l'augmentation de la précision.

### Arbre généralisé
<div style="display: grid; justify-content:around; grid-template-columns: 1fr 2fr;grid-gap: 30px;margin:auto;">
<div style="margin-top:auto; margin-bottom:auto;margin-right:30px">
<p>Avec hauteur = 1, les seuils, indices d'attribut et métriques sont identiques à celles obtenues dans la section 4.1.<br>
On constate de meilleurs résultats quand la hauteur = 2.</p>
</div>
<table style="text-align:center;">
  <tr>
    <th></th>
    <th>Weighted Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
  </tr>
  <tr>
    <td><b>Height = 1<b></td>
    <td>0.75800</td>
    <td>0.549020</td>
    <td>0.7000</td>
  </tr>
  <tr>
    <td><b>Height = 2</b></td>
    <td>0.79375</td>
    <td>0.557522</td>
    <td>0.7875</td>
  </tr>
    <tr>
    <td><b>Height = 3<b></td>
    <td>0.55700</td>
    <td>0.287582</td>
    <td>0.5500</td>
  </tr>
    <tr>
    <td><b>Height = 4<b></td>
    <td>0.43400</td>
    <td>0.193939</td>
    <td>0.4000</td>
  </tr>
</table>
</div>




