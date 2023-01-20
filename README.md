<div style="display: flex; background-color: RGB(255,114,0);" >
<div>
<img src="img/fraudeur_-_BNPP_PF_-_finale.jpg" width="300"/>
</div>

# PROJET - Comment démasquer les fraudeurs ? 
</div>

par BNP Paribas PF

Lien vers le challenge : https://challengedata.ens.fr/participants/challenges/104/


<div style="display: flex; background-color: Blue; padding: 15px;" >

## 1.Mission
</div>

L'objectif de ce challenge est de trouver la meilleure méthode pour transformer et agréger les données relatives au panier client d'un de nos parteneraires pour détecter les cas de fraude.

En utilisant ces données panier, les fraudeurs pourront être détectés, et ainsi refusés dans le futur.

<div style="display: flex; background-color: Green; padding: 7px;" >

### 1.1. Description des données
</div>

La base contient une liste d'achats effectués chez notre partenaire que nous avons financés. Les informations décrivent exclusivement le contenu du panier.

Pour chaque observation de la base, il y a 147 colonnes dont 144 peuvent être regroupées en 6 catégories (cf. Description des variables en entrée)

Le panier se décompose au maximum en 24 items. Par exemple, si un panier contient 3 items alors toutes les informations relatives à ces 3 items seront renseignées dans les colonnes item1, item2, item3, cash_price1, cash_price_2, cash_price3, make1, make2, make3, model1, model2, model3, goods_code1, goods_code2, goods_code3, Nbr_of_prod_purchas1, Nbr_of_prod_purchas2 et Nbr_of_prod_purchas3. Les variables restantes (celles avec un indice > 3) seront vides .

Un item correspond à un produit ou un regroupement de produits équivalents. Par exemple, si un panier contient 3 Iphones 14, alors ces 3 produits seront regroupés dans un seul item. Par contre, si le client achète 3 produits Iphone différents, alors nous considèrerons ici 3 items.

La variable Nb_of_items correspond au nombre d'items dans le panier, tandis que la somme des variables Nbr_of_fraud_purchas correspond au nombre de produits.

L’indicatrice fraud_flag permet de savoir si l’observation a été identifiée comme frauduleuse ou non.

<div style="display: flex; background-color: indigo;" >

##### 1.1.1. Description des variables en entrée (X)
</div>

|Variable 	|Description 	|Exemple    |
|-----------|---------------|----------------------------------------------------------------------|
|ID (Num) 	|Identifiant unique| 	1|
|item1 à item24 (Char)|Catégorie du bien de l'item 1 à 24 |Computer|
|cash_price1 à cash_price24 (Num)|Prix de l'item 1 à 24 |850|
|make1 à make24 (Char)|	Fabriquant de l'item 1 à 24|Apple|
|model1 à model24 (Char)|Description du modèle de l'item 1 à 24 |Apple Iphone XX|
|goods_code1 à goods_code24 (Char)|Code de l'item 1 à 24|2378284364|
|Nbr_of_prod_purchas1 à Nbr_of_prod_purchas24 (Num)|Nombre de produits dans l'item 1 à 24|2|
|Nb_of_items (Num) |Nombre total d'items|7|

<div style="display: flex; background-color: indigo;" >

##### 1.1.2. Description de la variable de sortie (Y)
</div>

|Variable|Description|
|-----------|---------------|
|ID (Num) |	Identifiant unique|
|fraud_flag (Num) |	Fraude = 1, Non Fraude = 0|

<div style="display: flex; background-color: indigo;" >

##### 1.1.3. Taille de la base
</div>

Taille : 115 988 observations, 147 colonnes.

Distribution de Y :

- Fraude (Y=1) : 1 681 observations
- Non Fraude (Y=0) : 114 307 observations

Le taux de fraude sur l'ensemble de la base est autour de 1.4%.

<div style="display: flex; background-color: Green; padding: 7px;" >

### 1.2. Echantillons 
</div>

La méthode d'échantillonnage appliquée est un tirage aléatoire simple sans remise. Ainsi, 80% de la base initiale a été utilisée pour générer l'échantillon de training et 20% pour l'échantillon de test.

<div style="display: flex; background-color: indigo;" >

#### 1.2.1. Echantillon d'entraînement
</div>

Taille : 92 790 observations, 147 colonnes. Distribution de Y_train :
- Fraude (Y=1) : 1 319 observations
- Non Fraude (Y=0) : 91 471 observations

<div style="display: flex; background-color: indigo;" >

#### 1.2.2. Echantillon de test
</div>
Taille : 23 198 observations, 147 colonnes. Distribution de Y_test :
- Fraude (Y=1) : 362 observations
- Non Fraude (Y=0) : 22 836 observations

<div style="display: flex; background-color: Green; padding: 7px;" >

## 3. Exploration des données (EDA) / Pré-processing
</div>

L'exploration des données et le pré-processing sont présentés ici : [bnp_paribas.ipynb](bnp_paribas.ipynb)

<div style="display: flex; background-color: Green; padding: 7px;" >

## 4. Benchmark 1
</div>

Le fichier présentant le 1er Benchmark :  [bnp_paribas_benchmarks_1_aleatoire.ipynb](bnp_paribas_benchmarks_1_aleatoire.ipynb)

<div style="display: flex; background-color: Green; padding: 7px;" >

## 5. Benchmark 2
</div>

Le fichier présentant le 2ème Benchmark :  [bnp_paribas_benchmarks_2.ipynb](bnp_paribas_benchmarks_2.ipynb)


<div style="display: flex; background-color: Green; padding: 7px;" >

## 6. Présentation des données
</div>

L'une de nos missions était de présenter les données sous Flask
