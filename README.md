# Projet_data_prevision_meteo
Projet complet de Science des données appliqué à la prévision météorologique en fonction d'anciennes données.



Commencer par lancer 'TraitementDesDonnées(df)' qui va supprimer les lignes vides, supprimer les colonnes inutiles et remplacer les valeurs abérantes.
Il prend un peu de temps à s'exécuter...

Appliquer 'df = df.reset_index(drop = True)' afin de réajuster les indices.

Lancer 'trouver_index_ville(df,0)' si on veut afficher les données de toutes les villes.
Cet algorithme remplace aussi les dates et noms des villes par des valeurs numérique.

Normalisation(df) normalise valeur par valeur; => très long et lourd 
Utiliser plutôt 'df_norm1 = (df - df.mean()) / (df.max() - df.min())' permet
de remettre toutes les valeurs sur une même échelle [-1,1]

Utiliser 'test = Test(df_norm1)' afin de créer des données de test
Utiliser 'X_train, X_test, y_train, y_test = test[0],test[1], test[2], test[3]' afin d'affecter
ces données et d'avoir nos jeux d'entrainement et de test

Utiliser 'RegressionLog(X_train, y_train)' afin d'effectuer une régression logistique

Utiliser 'EvaluationModele(X_test, y_test, X_train, y_train)' afin d'évaluer le modèle

Utiliser 'X_data = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
          y_data = df.iloc[:, [-1]]' 
          afin de séparer le jeu de données
          
Utiliser 'ValidationCroisée(X_data,y_data)' afin d'effectuer un test de validation croisée

