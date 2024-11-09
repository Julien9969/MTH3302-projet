# MTH3302-projet

# Projet Final - Prédiction de la Consommation en Carburant

## Description du Projet

Le projet final consiste en une évaluation en équipe (3 à 5 personnes) sous la forme d'une **compétition Kaggle privée** réservée à votre groupe. Ce projet a une pondération de **30 %** de la note globale du cours, et utilise les concepts étudiés pour résoudre un problème réel de prédiction en **apprentissage machine**.

Vous devrez :
1. Participer à la compétition Kaggle en ligne.
2. Rendre un **calepin Jupyter avec un noyau Julia**, qui documente en détail votre code.

Les équipes aux prédictions les plus précises recevront un bonus de **5, 3, et 2 points**. L’équipe gagnante pourra donc atteindre une note maximale de **35/30**.

Bien que des points bonus soient attribués pour la précision des prédictions, **ce n'est pas la seule manière de bien réussir le projet**. Votre créativité et la clarté de vos explications seront également prises en compte. Si votre modèle ne donne pas les résultats escomptés, expliquer votre démarche et proposer des améliorations potentielles vous permettront aussi de gagner des points.

## Consignes Générales

- **Constitution des équipes** : Chaque équipe doit comprendre entre **3 et 5 personnes**.
- **Compétition Kaggle** : Soumettez au moins une prédiction sur Kaggle.
- **Nom d'équipe** : Utilisez votre **numéro d'équipe** pour téléverser vos prédictions.
- **Rapport unique** : Remettez un seul fichier `.ipynb` par équipe, qui servira de rapport et permettra de reproduire vos meilleures prédictions.
- **Langage** : Utilisez le **langage Julia**.
- **Justification de la démarche** : La démarche doit être rigoureusement justifiée (consultez la grille de correction pour vous orienter).

## Sujet

### Problème à Résoudre

Vous devez prédire la consommation en carburant de voitures récentes. Le jeu de données contient des informations sur près de **400 véhicules** avec leur **consommation moyenne en L/100km** ainsi que d'autres caractéristiques. Votre objectif est de prédire la consommation de l'ensemble de test à partir de ces caractéristiques.

### Évaluation des Prédictions

Les performances de votre modèle seront mesurées avec la **racine de l'erreur quadratique moyenne (RMSE)** :

\[
\text{RMSE}(y^)=\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y^i - y_i)^2}
\]

### Format du Fichier de Prédictions

Le fichier de résultats à téléverser doit être un fichier **CSV** contenant deux colonnes :
1. **ID** : l'identifiant du véhicule.
2. **Consommation** : l'estimation de la consommation en L/100km.

Utilisez le calepin Jupyter de base fourni pour formater correctement ce fichier de prédictions.

### Limite de Soumissions

Vous disposez de **deux soumissions possibles par jour** sur Kaggle pour évaluer vos prédictions.

## Évaluation Finale

L’évaluation prend en compte :
- **La précision des prédictions** (points bonus pour les 3 premières équipes).
- **La créativité des solutions**.
- **La clarté des explications**.

Si vos prédictions ne sont pas suffisamment précises, **une analyse critique de votre modèle** et des **suggestions d'améliorations** pourront améliorer votre note.

## Rappel

Vous pouvez utiliser tous les concepts vus en cours. Si vous souhaitez appliquer des méthodes supplémentaires, vous devrez **justifier leur utilisation** et fournir des **citations appropriées**. L'utilisation de **réseaux de neurones n'est pas recommandée**.

Ce projet n’a pas pour objectif d'être stressant. Vous avez le temps de réfléchir, de discuter et de résoudre le problème avec **enthousiasme** !
