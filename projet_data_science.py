# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 08:46:14 2020

@author: Maxime
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from math import floor
from sklearn.preprocessing import StandardScaler
import numpy as np
from statistics import stdev
import time
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import model_selection



df = pd.read_csv('chemin vers le fichier csv')

df = df.replace(['No', 'Yes'], [0, 1])#Remplace les No par 0 et les Yes par 1

liste_nom_ville = df['Location']


    
def StrToInt(dataframe, nom_colonne):     # Associe un numéro à chaque ville
    label_encoder = LabelEncoder()
    npy = dataframe[nom_colonne].to_numpy()
    label_encoder.fit(npy)
    label_encoder.classes_
    dataframe[nom_colonne] = label_encoder.transform(npy)
        
def SuppValAberanteSup(dataframe, Nom_colonne, Valeur_maximale, Valeur_substitution, Nom_colonne_valeur_changée):#Remplace les valeur abérantes (c'est-à-dire les valeurs > Valeur_maximale).Il remplace ces valeurs par Valeur_substitution
    col = dataframe[Nom_colonne]
    col2 = col[col > Valeur_maximale]
    liste_index = col2.index
    for indice in liste_index: 
        dataframe[Nom_colonne_valeur_changée][indice] = Valeur_substitution
        
def SuppValAberanteInf(dataframe, Nom_colonne, Valeur_minimale, Valeur_substitution, Nom_colonne_valeur_changée):#Remplace les valeur abérantes (c'est-à-dire les valeurs < Valeur_minimale).Il remplace ces valeurs par Valeur_substitution
    col = dataframe[Nom_colonne]
    col2 = col[col < Valeur_minimale]
    liste_index = col2.index
    for indice in liste_index: 
        dataframe[Nom_colonne_valeur_changée][indice] = Valeur_substitution
    

def ch_moy(dataframe):#remplace les valeurs manquantes par la moyenne des autres valeurs
    L = [ 'MinTemp','Cloud9am','Cloud3pm',  'MaxTemp', 'Rainfall', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm','Pressure9am', 'Pressure3pm','RainToday',  'RainTomorrow' ]
    for k in L:
        remplace_moyenne(dataframe, k)
    return(df)
    
def SuppLignes(dataframe):#Supprime les lignes de RainToday sans valeur
    L = dataframe['RainToday'].isna()#tableau de booléens
    for k in range(len(L)):
        if(L[k] == True):#indice des lignes sans valeur pour la colonne RainToday
            dataframe.drop(k,0,inplace=True)#Supprime les lignes sans valeur
        

def remplace_moyenne(dataframe, colonne):
    mean = dataframe[colonne].mean()
    dataframe[colonne].fillna(mean, inplace = True)
    return(dataframe)
    
def supprime_colonne(dataframe):#supprimes les colonnes inutiles ou celles qui manquent de données
    del dataframe['WindGustDir']
    del dataframe['WindDir9am']
    del dataframe['WindDir3pm']
    del dataframe['WindSpeed9am']
    del dataframe['WindSpeed3pm']
    del dataframe['Evaporation']
    del dataframe['Sunshine']
    del dataframe['Temp9am']
    del dataframe['Temp3pm']
    del dataframe['RISK_MM']
    
    
    
def CorrectionRain(dataframe):#Si rain tomorrow = 1, vérifier que RainToday = 1 le jour suivant
    RainToday = dataframe['RainToday']
    RainTomorrow = dataframe['RainTomorrow']
    for i in range(len(RainTomorrow)):
        if(RainTomorrow[i] == 1 and RainToday[i+1] !=1):
            dataframe['RainToday'][i+1] = 1
    
def affichage_information(dataframe, a, b): #Affiche les informations sur une villes (température moyenne, humidité moyenne, pression moyenne)
    data = dataframe.iloc[a:b,:]
    Ville = liste_nom_ville[a]
    Temp_moyenne = (data['MinTemp'].mean() + data['MaxTemp'].mean())/2
    Humidité_moyenne = (data['Humidity9am'].mean() + data['Humidity3pm'].mean())/2
    Pression_moyenne = (data['Pressure9am'].mean() + data['Pressure3pm'].mean())/2
    print('') 
    print('') 
    print('----------------' + Ville + '----------------')
    print('A ' + Ville +', la température moyenne est de :' + str(floor(Temp_moyenne)) + '°C')
    print('A ' + Ville +', il y a une humidité moyenne de :' + str(floor(Humidité_moyenne)) + '%')
    print('A ' + Ville +', il y a une pression athmosphérique moyenne de :' + str(floor(Pression_moyenne)) + 'hPa')
    if(data.iloc[-1,-1] > 0.0):
        print("\033[31mIl risque de pleuvoir !\033[0m")
    print('') 
    print('') 


def trouver_index_ville(dataframe,a):#affiche les informations ville par ville (mettre 0 pour avoir toutes les données)
    StrToInt(dataframe, 'Location')#associe un entier à chaque ville
    StrToInt(dataframe, 'Date')#associe un entier à chaque date afin de pouvoir normaliser les données
    b = a
    l = dataframe['Location']
    while(l[b] == l[b+1] and b<140785):#tant que a = a+1, on a la même ville, on pourra les regrouper dans un même dataframe
        b = b + 1
    affichage_information(dataframe,a, b)
    if(b<140785):
        trouver_index_ville(dataframe, b+1)#représente le b de l'intervalle [a, b] dans lequel se trouve les informations d'une seule et même ville
    else:
        print("\033[31m               *Fin*\033[0m")
    
def TraitementDesDonnées(dataframe):#applique les autres algorithmes afin de rendre les données utilisables et pertinentes
    SuppLignes(dataframe)#Supprime les lignes où il manque des valeurs non remplacables (booléens)
    supprime_colonne(dataframe)#supprime les colonnes inutiles
    ch_moy(dataframe)#Complète les valeurs manquantes en insérant la moyenne des autres données
    SuppValAberanteSup(dataframe,'MaxTemp', 50.0, df['MaxTemp'].mean(), 'MaxTemp')#remplace les températures maximales > 50° par la moyenne des autres températures
    SuppValAberanteInf(dataframe,'MaxTemp', 5.0, df['MaxTemp'].mean(), 'MaxTemp')#Remplace les températures maximales inférieures à 5° par la moyenne des autres températures
    SuppValAberanteSup(dataframe,'Rainfall', 150.0, df['Rainfall'].mean(), 'Rainfall')#Remplace les données de relevé de pluie > 150.0 mm par la moyenne des autres valeurs
    SuppValAberanteSup(dataframe,'Cloud3pm', 8.0, df['Cloud3pm'].mean(), 'Cloud3pm')#remplace les valeurs de Cloud > 8 (impossible)
    SuppValAberanteSup(dataframe,'Cloud9am', 8.0, df['Cloud9am'].mean(), 'Cloud9am')#remplace les valeurs de Cloud > 8 (impossible)
    
    
    
def correlation(dataframe):#Trouve les colonnes les plus susceptibles d'être corrélées
    ListeColonnes = [ 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
       'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
       'Cloud3pm', 'RainToday', 'RainTomorrow']
    L = []#Liste des variables potentiellement corrélées
    for col1 in ListeColonnes :
        l1 = []
        for col2 in ListeColonnes :
            coef_corr = df[col1].corr(df[col2])
            if (coef_corr > 0.5 and coef_corr < 1.0):#forte probabilité de corrélation positive
                l1.append(col1)
                l1.append(col2)
                print("Le coef de corrélation de " + col1 + " et de " + col2 + " est " + str(coef_corr))
                L.append(l1)
            if(coef_corr < -0.5 and coef_corr>-1.0):#forte probabilité de corrélation négative
                l1.append(col1)
                l1.append(col2)
                print("Le coef de corrélation de " + col1 + " et de " + col2 + " est " + str(coef_corr))
                L.append(l1)    
    return(L)
    
def Normalisation(dataframe):
    start_time = time.time()
    ListeColonnes = [ 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
       'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
       'Cloud3pm', 'RainToday', 'RISK_MM', 'RainTomorrow','Date'] 
    x = 0

    while x <= 14 : 
        for colonne in ListeColonnes :
            l = dataframe[colonne]
            mean = l.mean()
            ecart_type = stdev(l)
            for k in range(len(l)) :
                dataframe[colonne][k] = (l[k] - mean)/ecart_type
            print(colonne)
            x = x + 1
    print("Temps d execution : %s secondes ---" % (time.time() - start_time))#Environ 5 min d'execution en moyenne 

def ScatterMatrice(df):#Affiche la scatter_matrix
    x = df.iloc[: , [2,3,4,5,6,7,8,9,10,11,12, 13]]#Sélection des colonnes potentiellement corrélées
    scatter_matrix(x, alpha=0.2, figsize=(6, 6), diagonal='kde')
    
def NormalisationDesDonnées(x):#normalise les données pour les mettre à la même échelle (mettre x = df.values !)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(x)
    df = scaled_data
    return (df)


def Test(dataframe):
    
    X_data = dataframe.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
    y_data = dataframe.iloc[:, [-1]]
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)
    proportion_train = (len(X_train)/len(dataframe))*100
    proportion_test = (len(X_test)/len(dataframe))*100
    print("Il y a" + str(floor(proportion_train)) + "% de données d'entrainement et " + str(floor(proportion_test)) + "% de données de test")
    """
    X_train = X_train.sort_index(axis = 0, ascending = True)
    X_test = X_test.sort_index(axis = 0, ascending = True)
    y_test = y_test.sort_index(axis = 0, ascending = True)
    y_train = y_train.sort_index(axis = 0, ascending = True)
    """
    StrToInt(X_train, 'Date')
    StrToInt(X_train, 'Location')
    StrToInt(X_test, 'Date')
    StrToInt(X_test, 'Location')
    return(X_train, X_test, y_train, y_test)
    
def RegressionLog(X_train, y_train):
    logisticregression = LogisticRegression()
    modele = logisticregression.fit(X_train, y_train)
    print(modele.coef_,modele.intercept_)
    
def EvaluationModele(X_test, y_test, X_train, y_train):
    logisticregression = LogisticRegression()
    modele = logisticregression.fit(X_train, y_train)
    y_pred = modele.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)#Matrice de confusion
    print(cm)
    acc = metrics.accuracy_score(y_test, y_pred)#taux de succès
    print("Taux de succès :" + str(acc))
    err = 1.0 - acc#taux d'erreur
    print("taux erreur :" + str(err))
    se = metrics.recall_score(y_test, y_pred)#sensibilité
    print("Sensibilité :" +str(se))
    
def ValidationCroisée(X,y):
    logisticregression = LogisticRegression()
    succes = model_selection.cross_val_score(logisticregression,X,y,cv=10,scoring='accuracy')
    print(succes)
    print(succes.mean())
    
    
"""---Exemple1---SansNormalisation--
>>>test = Test(df)
[...]
>>>X_train, X_test, y_train, y_test = test[0],test[1], test[2], test[3]
>>>RegressionLog(X_train, y_train)

[[ 1.33148444e-05  3.50597901e-04  3.50035948e-02 -4.81159390e-02
   5.88763253e-04  7.99096996e-03  1.39455581e-02  9.81417398e-03
  -7.29579499e-03 -4.00789045e-04  7.98980048e-03 -8.98007025e-02
   3.60320947e-01  4.49365304e+00  1.09328530e+01]] [0.00530416]

>>>EvaluationModele(X_test, y_test, X_train, y_train)

[[22638     0]
 [    0 12559]]
Taux de succès :1.0
taux erreur :0.0
Sensibilité :1.0

>>>X_data = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
y_data = df.iloc[:, [-1]]

>>>ValidationCroisée(X_data,y_data)

[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
1.0


"""


"""---Exemple2---AvecNormalisationManuelle

>>>Normalisation(df)
[...]
>>>test = Test(df)

Il y a74% de données d'entrainement et 25% de données de test
[...]
>>>X_train, X_test, y_train, y_test = test[0],test[1], test[2], test[3]
>>>RegressionLog(X_train, y_train)

[[-7.00175130e-01 -3.68785120e-01 -8.16585755e-03 -3.27838163e-02
   3.63836469e-02  1.64268087e-01  1.45242090e-01  3.60005197e-01
  -1.78005891e-02 -9.72587461e-02  7.34240884e-02  1.50651763e-01
   1.56175473e-01  3.13138597e+00  1.50367962e+01]] [-5.44395862]

>>>EvaluationModele(X_test, y_test, X_train, y_train)

[[22706     0]
 [    0 12491]]
Taux de succès :1.0
taux erreur :0.0
Sensibilité :1.0

>>>X_data = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
y_data = df.iloc[:, [-1]]

>>>ValidationCroisée(X_data,y_data)

[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
1.0

"""

"""---Exemple2---AvecNormalisationEchelle [-1,1]

>>>df_norm1 = (df - df.mean()) / (df.max() - df.min())
[...]
>>>test = Test(df)

Il y a 74% de données d'entrainement et 25% de données de test
[...]
>>>X_train, X_test, y_train, y_test = test[0],test[1], test[2], test[3]
>>>RegressionLog(X_train, y_train)

[[-1.06291367e-05  1.47108531e-03  1.43013311e-02 -2.85985082e-02
  -1.51640039e-02  3.16384236e-03  1.22905847e-02  7.72299603e-03
  -2.78135029e-02  2.01889138e-02  1.03967449e-01  8.73969726e-02
   3.65222154e-01  3.97600117e+00  9.03953447e+00]] [0.00545183]

>>>EvaluationModele(X_test, y_test, X_train, y_train)

[[22706     0]
 [    0 12491]]
Taux de succès :1.0
taux erreur :0.0
Sensibilité :1.0

>>>X_data = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
y_data = df.iloc[:, [-1]]

>>>ValidationCroisée(X_data,y_data)

[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
1.0

"""
