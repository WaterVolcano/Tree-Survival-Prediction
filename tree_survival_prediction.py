# -*- coding: utf-8 -*-
"""Tree_survival_prediction.ipynb

# Citire set de date
"""

import pandas as pd

# importarea setului de date
from google.colab import files
incarcare = files.upload()

df = pd.read_csv("Tree_Data.csv")

df.head()

df.keys()

"""# Copierea coloanelor de interes intr-o variabila noua"""

df_control = df[["Species","Light_ISF","Light_Cat","Soil","Sterile","Conspecific","Myco","SoilMyco","PlantDate", "Event"]]
df_control.head()

df2 = df_control.copy()
df2.head()

df2.info()

"""# Lable encoding"""

#schimbam denumirile cu cifre pentru a face procesarea de catre calculator mai usoara
def specii_numeric(Species):
    if Species == "Acer saccharum" :
        return 1
    elif Species == "Prunus serotina":
        return 2
    elif Species == "Quercus alba":
        return 3
    elif Species == "Quercus rubra":
        return 4
def categorie_de_lumina_numeric(Light_Cat):
    if Light_Cat == "Low" :
        return 1
    elif Light_Cat == "Med":
        return 2
    elif Light_Cat == "High":
        return 3
def sol_numeric(Soil):
    if Soil == "Acer rubrum" :
        return 1
    elif Soil == "Acer saccharum":
        return 2
    elif Soil == "Populus grandidentata":
        return 3
    elif Soil == "Prunus serotina":
        return 4
    elif Soil == "Quercus alba":
        return 5
    elif Soil == "Quercus rubra":
        return 6
    elif Soil == "Sterile":
        return 7
def steril_numeric(Sterile):
    if Sterile == "Non-Sterile" :
        return 1
    elif Sterile == "Sterile":
        return 2
def conspecific_numeric(Conspecific):
    if Conspecific == "Conspecific" :
        return 1
    elif Conspecific == "Heterospecific":
        return 2
    elif Conspecific == "Sterilized":
        return 3
def Myco_numeric(Myco):
    if Myco == "AMF" :
        return 1
    elif Myco == "EMF":
        return 2
def SoilMyco_numeric(SoilMyco):
    if SoilMyco == "AMF" :
        return 1
    elif SoilMyco == "EMF":
        return 2
    elif SoilMyco == "Sterile":
        return 3

df2.loc[:, 'Species'] = df2['Species'].apply(specii_numeric)
df2.loc[:, 'Light_Cat'] = df2['Light_Cat'].apply(categorie_de_lumina_numeric)
df2.loc[:, 'Soil'] = df2['Soil'].apply(sol_numeric)
df2.loc[:, 'Sterile'] = df2['Sterile'].apply(steril_numeric)
df2.loc[:, 'Conspecific'] = df2['Conspecific'].apply(conspecific_numeric)
df2.loc[:, 'Myco'] = df2['Myco'].apply(Myco_numeric)
df2.loc[:, 'SoilMyco'] = df2['SoilMyco'].apply(SoilMyco_numeric)
df2

df2.head()

# am ales sa folosim o alta metoda pentru coloana 'PlantDate' pentru a arata si aceasta posibilitate
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

df2['PlantDate']= label_encoder.fit_transform(df2['PlantDate'])
df2['PlantDate'].unique()

df2.head()

"""# Eliminarea valorile lipsa"""

df2.info()

df2.shape

#Verificam care coloane contin valori lipsa, si cate valori lipsa sunt
df2.isna().sum()

linia_control = df2.iloc[2781] # acesta linia nu contine date in coloana event, asa ca o vom folosi pentru predictie

linia_control

# Stergem randul care contine o valoare lipsa in coloana Event
df2 = df2.dropna(subset = 'Event')

"""# EDA"""

print(df.isna().sum())

df2.describe()

!pip install matplotlib pandas scikit-learn

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

hue_order = [1, 0]
sns.pairplot(df2, hue='Event', hue_order = hue_order)

#EDA (Analiza exploratoare a datelor)
sns.countplot(x="Event", data = df2)

# Comparatie intre distributiile celor 4 specii
sns.countplot(x="Species", data = df)

# Distributia originilor solurilor folosite in experiment
plt.figure(figsize=(15, 5))
sns.countplot(x="Soil", data = df)

# Distributia conspecificitatii solurilor intre specia plantata si originea solului
sns.countplot(x="Conspecific", data = df)

# Matricea de corelari intre diferitele elemente ale datasetului
plt.figure(figsize=(8,8))
sns.heatmap(df2.drop("Event", axis = 1).corr(), cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':10}, cmap='RdPu')
plt.show()

# Distributia densitatilor valorilor de lumina codate color dupa rezultatul experimentului (kernel density estimate)
# se pot observa valori care au afectat pozitiv si valori care au afectat negativ rezultatul

import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.kdeplot(data=df2, x="Light_ISF", hue="Event", hue_order = hue_order, ax=axes[0])
axes[0].set_title("Distributia nivelurilor de lumina colorate conform rezultatului")
axes[0].set_ylim(0, 20)

sns.kdeplot(data=df2, x="Light_ISF", ax=axes[1])
axes[1].set_title("Distributia nivelurilor de lumina indiferent de rezultat")
axes[1].set_ylim(0, 21)

plt.tight_layout()
plt.show()

# Corelatia dintre specie si rezultat

plt.figure(figsize=(16, 7))
plt.subplot(1,2,2)
sns.violinplot(x="Event", y="Species", data=df, color = "green")
plt.show()

# Numararea aparitiilor diferitelor valori in anumite coloane de interes

# Specii, 4 la numar
print(df['Species'].value_counts(sort = True))
print()

# Nivelul de lumina
print(df['Light_Cat'].value_counts(sort = True))
print()

# Specia de la care a fost prelevat solul pentru experiment, 6 la numar, cele 4 de mai sus + alte doua
print(df['Soil'].value_counts(sort = True))
print()

# Daca solul a fost sau nu sterilizat
print(df['Sterile'].value_counts(sort = True))
print()

# Daca solul a fost folosit la aceeasi specie de la care a fost prelevat (Conspecific), daca a fost folosit la alta specie (Heterospecific), sau daca a fost steril (Sterilized)
# Conform studiului (https://datadryad.org/stash/dataset/doi:10.5061/dryad.xd2547dpw), tipul de sol sterilizat provine de la aceeasi specie la care a fost si folosit, deci poate fi considerat tot conspecific
print(df['Conspecific'].value_counts(sort = True))
print()

# Tipul de micorizare al rasadului: AMF (Arbuscular Micorizal Fungi) sau EMF (Ecto Micorizal Fungi)
# Conform studiului, nivelurile de AMF si EMF au fost masurate la 3 saptamani dupa inceperea experimentului
print(df['Myco'].value_counts(sort = True)) # Se observa ca in toate cazurile in care rasadul a prezentat o micorizare arbusculara, ectomicorizarea nu are loc (NA)
print()

# Tipul de micorizare al solului: AMF sau EMF
print(df['SoilMyco'].value_counts(sort = True))

# Se poate observa ca raportul dintre cele doua valori din coloana Event se ridica foarte mul in favoarea 0,
# fapt ce ne impinge sa credem ca intre valorile NA din coloana EMF si rezultatul experimentului exista o relatie de corelare, daca nu chiar de cauzalitate

df_control

"""# Normalizare"""

normal_df11 = df2.copy()
column = [['Species', 'Light_ISF',	'Light_Cat',	'Soil',	'Sterile',	'Conspecific',	'Myco',	'SoilMyco','PlantDate',	'Event'  ]]
for col in column:
  normal_df11[col] = (normal_df11[col] - normal_df11[col].min()) / (normal_df11[col].max() - normal_df11[col].min())
normal_df11

df2 = normal_df11.copy()
df2.head()

column = [['Species', 'Light_ISF',	'Light_Cat','Soil',	'Sterile',	'Conspecific',	'Myco',	'SoilMyco',	'PlantDate','Event'  ]]
normal_df2 = (df2 - df2.min()) / (df2.max() - df2.min())
for col in column:
    normal_df2[col] = df2[col]
normal_df2

"""# PCA"""

# df2

# from sklearn.preprocessing import StandardScaler
# %matplotlib inline

# x = df2[["Species",	"Light_ISF",	"Light_Cat",	"Soil",	"Sterile",	"Conspecific",	"SoilMyco"]].values
# #x = df2.loc[:, 7].values
# y = df2.loc[:,['Event']].values
# x = StandardScaler().fit_transform(x)
# pd.DataFrame(data = x, columns = df2[["Species",	"Light_ISF",	"Light_Cat",	"Soil",	"Sterile",	"Conspecific",	"SoilMyco"]]).head()

# X_PCA = df2[["Species",	"Light_ISF",	"Light_Cat",	"Soil",	"Sterile",	"Conspecific",	"SoilMyco"]] #train
# y_PCA = df2["Event"] #test

# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)

# principalComponents = pca.fit_transform(X_PCA)

# principalDf = pd.DataFrame(data = principalComponents
#              , columns = ['principal component 1', 'principal component 2'])

# finalDf = pd.concat([principalDf, df2[['Event']]], axis = 1)

# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)

# targets = ['1','0']
# colors = ['r', 'g', 'b']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['Event'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import StandardScaler
# df_scalat = StandardScaler().fit_transform(df2.T)

# pca = PCA()
# pca.fit(df_scalat)
# pca_data = pca.transform(df_scalat)

# per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
# #labels = ['keys' + str(x) for x in range(1, len(per_var)+1)]
# column_names = df2.columns
# plt.figure(figsize = (24,12))
# plt.bar( x = range(1, len(per_var)+1), height = per_var, tick_label = column_names)
# plt.ylabel("Procentaj")
# plt.xlabel("Component principal")
# plt.title("Plot")
# plt.show()

# pca_df2.shape()

# #pca_df = df2.copy()
# pca_df1 = df2[["Species",	"Light_ISF",	"Light_Cat",	"Soil",	"Sterile",	"Conspecific",	"SoilMyco"]]
# pca_df2 = df2["Event"]
# plt.scatter(pca_df1,["Event"])
# plt.ylabel('PC1-{0}%'.format(per_var[0]))
# plt.xlabel('PC2-{0}%'.format(per_var[1]))
# plt.title("Plot")

# for sample in pca_df.index:
#   plt.annotate(sample, (pca_df.PC1.loc[sample],pca_df.PC2.loc[sample]))
# plt.show()

# X_PCA = df2[["Species",	"Light_ISF",	"Light_Cat",	"Soil",	"Sterile",	"Conspecific",	"SoilMyco"]] #train
# y_PCA = df2["Event"] #test

# from sklearn.model_selection import train_test_split
# X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = train_test_split(X_PCA, y_PCA, test_size=0.2)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()

# X_train_PCA = sc.fit_transform(X_train_PCA)
# X_test_PCA = sc.transform(X_test_PCA)

# # Applying PCA function on training
# # and testing set of X component
# from sklearn.decomposition import PCA

# pca = PCA(n_components = 2)

# X_train_PCA = pca.fit_transform(X_train_PCA)
# X_test_PCA = pca.transform(X_test_PCA)

# explained_variance = pca.explained_variance_ratio_

# # Fitting Logistic Regression To the training set
# from sklearn.linear_model import LogisticRegression

# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X_train_PCA, y_train_PCA)

# # Predicting the test set result using
# # predict function under LogisticRegression
# y_pred_PCA = classifier.predict(X_test_PCA)

# # making confusion matrix between
# #  test set of Y and predicted value.
# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test_PCA, y_pred_PCA)

# cm

# # Predicting the training set
# # result through scatter plot
# from matplotlib.colors import ListedColormap

# X_set, y_set = X_train_PCA, y_train_PCA
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
# 					stop = X_set[:, 0].max() + 1, step = 0.01),
# 					np.arange(start = X_set[:, 1].min() - 1,
# 					stop = X_set[:, 1].max() + 1, step = 0.01))

# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
# 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
# 			cmap = ListedColormap(('yellow', 'white', 'aquamarine')))

# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())

# for i, j in enumerate(np.unique(y_set)):
# 	plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
# 				c = ListedColormap(('red', 'green', 'blue'))(i), label = j)

# plt.title('Logistic Regression (Training set)')
# plt.xlabel('PC1') # for Xlabel
# plt.ylabel('PC2') # for Ylabel
# plt.legend() # to show legend

# # show scatter plot
# plt.show()

# # Visualising the Test set results through scatter plot
# from matplotlib.colors import ListedColormap

# X_set, y_set = X_test_PCA, y_test_PCA

# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
# 					stop = X_set[:, 0].max() + 1, step = 0.01),
# 					np.arange(start = X_set[:, 1].min() - 1,
# 					stop = X_set[:, 1].max() + 1, step = 0.01))

# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
# 			X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
# 			cmap = ListedColormap(('yellow', 'white', 'aquamarine')))

# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())

# for i, j in enumerate(np.unique(y_set)):
# 	plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
# 				c = ListedColormap(('red', 'green', 'blue'))(i), label = j)

# # title for scatter plot
# plt.title('Logistic Regression (Test set)')
# plt.xlabel('PC1') # for Xlabel
# plt.ylabel('PC2') # for Ylabel
# plt.legend()

# # show scatter plot
# plt.show()

# print('Explained variation per principal component: {}'.format(pca_Tree.explained_variance_ratio_))

# import matplotlib.pyplot as plt
# %matplotlib inline

# plt.figure()
# plt.figure(figsize=(10,10))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=14)
# plt.xlabel('Principal Component - 1',fontsize=20)
# plt.ylabel('Principal Component - 2',fontsize=20)
# plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
# targets = ['Benign', 'Malignant']
# colors = ['r', 'g']
# for target, color in zip(targets,colors):
#     indicesToKeep = principal_Tree_Df['] == target
#     plt.scatter(principal_Tree_Df.loc[indicesToKeep, 'principal component 1']
#                , principal_Tree_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

# plt.legend(targets,prop={'size': 15})

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# # Load your dataset
# #data = pd.read_csv('your_dataset.csv')

# # Preprocess your data (e.g., handle missing values, scale features)
# # Standardize the features
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(df2)

# # Perform PCA
# pca = PCA(n_components=2)  # specify the number of components you want to retain
# principal_components = pca.fit_transform(data_scaled)

# # Create a DataFrame to store the principal components
# principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# # Optional: Explore principal components and their loadings
# print(pca.explained_variance_ratio_)  # Variance explained by each principal component
# print(pca.components_)  # Principal axes in feature space, representing the directions of maximum variance

# # You can use principal_df for further analysis or visualization

# import matplotlib.pyplot as plt

# # Scatter plot of the first two principal components
# plt.figure(figsize=(8, 6))
# plt.scatter(principal_df['PC1'], principal_df['PC2'], alpha=0.5)
# plt.title('PCA Plot of Data')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid(True)
# plt.show()

"""# Standardizare"""

#standardizare
#standartized_df=(df2-df2.mean())/df2.std()  # Standardizare = (variabila-mean)/ standard deviation a variabilei
#standartized_df

"""---

---


# SVM

## Cu kernel liniar

impartim data setul in valori de test si de antrenament
"""

# from sklearn.model_selection import train_test_split
# X_SVM_liniar = df2[[ 'Species', 'Light_ISF',	'Soil',	'Sterile',	'Conspecific',	'Myco',	'SoilMyco',	'PlantDate']] #caracteristici
# y_SVM_liniar = df2["Event"] #target

# X_train_SVM_liniar, X_test_SVM_liniar, y_train_SVM_liniar, y_test_SVM_liniar = train_test_split(X_SVM_liniar, y_SVM_liniar, test_size=0.3, random_state=42)

# from sklearn import svm
# clf = svm.SVC(kernel='linear') # Linear Kernel
# clf.fit(X_train_SVM_liniar, y_train_SVM_liniar)
# y_pred_SVM_liniar = clf.predict(X_test_SVM_liniar)

# cv_scores = cross_val_score(clf, X_train_SVM_liniar, y_train_SVM_liniar, cv=15)

# print("Scoruri cross-validation:", cv_scores)
# print("Media:", cv_scores.mean())

# from sklearn import metrics
# print("Accuracy:",metrics.accuracy_score(y_test_SVM_liniar, y_pred_SVM_liniar))
# print("Precission:",metrics.precision_score(y_test_SVM_liniar, y_pred_SVM_liniar))
# print("f1:",metrics.f1_score(y_test_SVM_liniar, y_pred_SVM_liniar))
# print("confusion_matrix:",'\n',metrics. confusion_matrix(y_test_SVM_liniar, y_pred_SVM_liniar))

"""## Cu Kernel Polinormial"""

# from sklearn.model_selection import train_test_split
# X_SVM_poli = df2[['Species', 'Light_ISF',	'Soil',	'Sterile',	'Conspecific',	'Myco',	'SoilMyco',	'PlantDate']]
# y_SVM_poli = df2["Event"]

# X_train_SVM_poli, X_test_SVM_poli, y_train_SVM_poli, y_test_SVM_poli = train_test_split(X_SVM_poli, y_SVM_poli, test_size=0.3,  random_state=42)

# from sklearn import svm
# clf = svm.SVC(kernel='poly', degree=4, C=1.0, gamma='auto') # Polinomial Kernel
# clf.fit(X_train_SVM_poli, y_train_SVM_poli)
# y_pred_SVM_poli = clf.predict(X_test_SVM_poli)

# cv_scores = cross_val_score(clf, X_train_SVM_poli, y_train_SVM_poli, cv=15)

# print("Scoruri cross-validation:", cv_scores)
# print("Media:", cv_scores.mean())

# from sklearn import metrics
# print("Accuracy:",metrics.accuracy_score(y_test_SVM_poli, y_pred_SVM_poli))
# print("Precission:",metrics.precision_score(y_test_SVM_poli, y_pred_SVM_poli))
# print("f1:",metrics.f1_score(y_test_SVM_poli, y_pred_SVM_poli))
# print("confusion_matrix:",'\n',metrics. confusion_matrix(y_test_SVM_poli, y_pred_SVM_poli))

"""## RBF Kernel"""

# from sklearn.model_selection import train_test_split
# X_SVM_RBF = df2[['Species', 'Light_ISF',	'Soil',	'Sterile',	'Conspecific',	'Myco',	'SoilMyco',	'PlantDate']]
# y_SVM_RBF = df2["Event"]

# X_train_SVM_RBF, X_test_SVM_RBF, y_train_SVM_RBF, y_test_SVM_RBF = train_test_split(X_SVM_RBF, y_SVM_RBF, test_size=0.3,  random_state=42)

# from sklearn import svm
# clf = svm.SVC(kernel='rbf', C=100, gamma='auto') # RBF Kernel
# clf.fit(X_train_SVM_RBF, y_train_SVM_RBF)
# y_pred_SVM_RBF = clf.predict(X_test_SVM_RBF)

# cv_scores = cross_val_score(clf, X_train_SVM_RBF, y_train_SVM_RBF, cv=15)

# print("Scoruri cross-validation:", cv_scores)
# print("Media:", cv_scores.mean())

# from sklearn import metrics
# print("Accuracy:",metrics.accuracy_score(y_test_SVM_RBF, y_pred_SVM_RBF))
# print("Precission:",metrics.precision_score(y_test_SVM_RBF, y_pred_SVM_RBF))
# print("f1:",metrics.f1_score(y_test_SVM_RBF, y_pred_SVM_RBF))
# print("confusion_matrix:",'\n',metrics. confusion_matrix(y_test_SVM_RBF, y_pred_SVM_RBF))

"""## Sigmoid Kernel"""

# from sklearn.model_selection import train_test_split
# X_SVM_s = df2[['Species', 'Light_ISF',	'Soil',	'Sterile',	'Conspecific',	'Myco',	'SoilMyco',	'PlantDate']] #train
# y_SVM_s = df2["Event"] #test
# X_train_SVM_s, X_test_SVM_s, y_train_SVM_s, y_test_SVM_s = train_test_split(X_SVM_s, y_SVM_s, test_size=0.2, random_state=42)

# from sklearn import svm
# clf = svm.SVC(kernel='sigmoid', C=0.001) # Sigmoid Kernel
# clf.fit(X_train_SVM_s, y_train_SVM_s)
# y_pred_SVM_s = clf.predict(X_test_SVM_s)

# cv_scores = cross_val_score(clf, X_train_SVM_s, y_train_SVM_s, cv=15)

# print("Scoruri cross-validation:", cv_scores)
# print("Media:", cv_scores.mean())

# from sklearn import metrics
# print("Accuracy:",metrics.accuracy_score(y_test_SVM_s, y_pred_SVM_s))
# print("Precission:",metrics.precision_score(y_test_SVM_s, y_pred_SVM_s))
# print("f1:",metrics.f1_score(y_test_SVM_s, y_pred_SVM_s))
# print("confusion_matrix:",'\n',metrics. confusion_matrix(y_test_SVM_s, y_pred_SVM_s))

"""# K-nn"""

# df2.info()

# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import f1_score

# print(df2.head())
# print("Lungime set:", len(df2))

# X_Knn = df2[['Species', 'Light_ISF',	'Soil',	'Sterile',	'Conspecific',	'Myco',	'SoilMyco',	'PlantDate']]
# y_Knn = df2["Event"]

# X_train_Knn, X_test_Knn, y_train_Knn, y_test_Knn = train_test_split(X_Knn, y_Knn, test_size=0.2,  random_state=42)

# import math
# lungime = len(df2)
# print(math.sqrt(lungime))   # ===> vom folosi K = 53

# clasificator = KNeighborsClassifier(n_neighbors=53, p=2, metric = 'euclidean')   # p=2 pentru ca avem 2 valori posibile de prezis: 0 sau 1

# clasificator.fit (X_train_Knn,y_train_Knn)
# y_pred_Knn = clasificator.predict(X_test_Knn)

# cv_scores = cross_val_score(clasificator, X_train_Knn, y_train_Knn, cv=15)

# print("Scoruri cross-validation:", cv_scores)
# print("Media:", cv_scores.mean())

# print("Accuracy:",metrics.accuracy_score(y_test_Knn, y_pred_Knn))
# print("Precission:",metrics.precision_score(y_test_Knn, y_pred_Knn))
# print("f1:",metrics.f1_score(y_test_Knn, y_pred_Knn))
# print("confusion_matrix:",'\n',metrics. confusion_matrix(y_test_Knn, y_pred_Knn))

# #pastram 25%
# import random
# total_randuri_de_sters = df2.shape[0]
# randuri_de_sters = int(total_randuri_de_sters * 0.75)
# randuri_de_pastrat = random.sample(range(total_randuri_de_sters), total_randuri_de_sters - randuri_de_sters)
# df25 = df2.iloc[randuri_de_pastrat]
# df25.head()

# X_Knn25 =df25[['Species', 'Light_ISF',	'Soil',	'Sterile',	'Conspecific',	'Myco',	'SoilMyco',	'PlantDate' ]]#train
# y_Knn25 = df25["Event"] #test
# X_train_Knn25, X_test_Knn25, y_train_Knn25, y_test_Knn25 = train_test_split(X_Knn25, y_Knn25, test_size=0.2,  random_state=42)

# import math
# lungime = len(df25)
# print(math.sqrt(lungime))   # ===> vom folosi K =27

# clasificator25 = KNeighborsClassifier(n_neighbors=27, p=2, metric = 'euclidean')   # p=2 pentru ca avem 2 valori posibile de prezis: 0 sau 1
# clasificator25.fit (X_train_Knn25,y_train_Knn25)
# y_pred_Knn25 = clasificator25.predict(X_test_Knn25)
# print("Accuracy:",metrics.accuracy_score(y_test_Knn25, y_pred_Knn25))
# print("Precission:",metrics.precision_score(y_test_Knn25, y_pred_Knn25))
# print("f1:",metrics.f1_score(y_test_Knn25, y_pred_Knn25))
# print("confusion_matrix:",'\n',metrics. confusion_matrix(y_test_Knn25, y_pred_Knn25))

# cv_scores = cross_val_score(clasificator25, X_train_Knn25, y_train_Knn25, cv=15)

# print("Scoruri cross-validation:", cv_scores)
# print("Media:", cv_scores.mean())

"""# Decision Tree"""

# from sklearn.tree import DecisionTreeClassifier

# X_tree = df2[['Species', 'Light_ISF',	'Soil',	'Sterile',	'Conspecific',	'Myco',	'SoilMyco',	'PlantDate']]
# y_tree = df2["Event"]
# X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y_tree, test_size=0.2, random_state=42)

# from sklearn.model_selection import cross_val_score
# model_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
# scoruri_entropie = cross_val_score(model_entropy, X_tree, y_tree, cv = 25)    # cv = folduri - nr de cutii in care aleg sa impart datele respective
# print(f'Scorurile pentru Information Gain / Entropie: {scoruri_entropie}')

# clasificare = DecisionTreeClassifier()
# clasificare.fit(X_train_tree, y_train_tree)
# y_pred_tree = clasificare.predict(X_test_tree)

# cv_scores = cross_val_score(clasificare, X_train_tree, y_train_tree, cv=15)

# print("Scoruri cross-validation:", cv_scores)
# print("Media:", cv_scores.mean())

# from sklearn import metrics
# print("Accuracy:",metrics.accuracy_score(y_test_tree, y_pred_tree))
# print("Precission:",metrics.precision_score(y_test_tree, y_pred_tree))
# print("f1:",metrics.f1_score(y_test_tree, y_pred_tree))
# print("confusion_matrix:",'\n',metrics. confusion_matrix(y_test_tree, y_pred_tree))

"""# Random forest"""

# from sklearn.ensemble import RandomForestClassifier
# X_forest = df2[['Species', 'Light_ISF',	'Soil',	'Sterile',	'Conspecific',	'Myco',	'SoilMyco',	'PlantDate']]
# y_forest = df2["Event"]
# X_train_forest, X_test_forest, y_train_forest, y_test_forest = train_test_split(X_forest, y_forest, test_size=0.2, random_state=42)

# rf_classifier = RandomForestClassifier(n_estimators = 500, random_state=42)
# rf_classifier.fit(X_train_forest, y_train_forest)
# y_pred_forest = rf_classifier.predict(X_test_forest)

# cv_scores = cross_val_score(rf_classifier, X_train_forest, y_train_forest, cv=15)

# print("Scoruri cross-validation:", cv_scores)
# print("Media:", cv_scores.mean())

# from sklearn import metrics
# print("Accuracy:",metrics.accuracy_score(y_test_forest, y_pred_forest))
# print("Precission:",metrics.precision_score(y_test_forest, y_pred_forest))
# print("f1:",metrics.f1_score(y_test_forest, y_pred_forest))
# print("confusion_matrix:",'\n',metrics. confusion_matrix(y_test_forest, y_pred_forest))

"""# Logistic Regression"""

# X_logi = df2[['Species', 'Light_ISF',	'Soil',	'Sterile',	'Conspecific',	'Myco',	'SoilMyco',	'PlantDate']]
# y_logi = df2["Event"]

# X_train_logi, X_test_logi, y_train_logi, y_test_logi = train_test_split(X_logi, y_logi, test_size=0.2, random_state=42)

# from sklearn.linear_model import LogisticRegression
# clasificator = LogisticRegression()

# clasificator.fit (X_train_logi, y_train_logi)
# y_pred_logi = clasificator.predict(X_test_logi)

# cv_scores = cross_val_score(clasificator, X_train_logi, y_train_logi, cv=15)

# print("Scoruri cross-validation:", cv_scores)
# print("Media:", cv_scores.mean())

# print("Accuracy:",metrics.accuracy_score(y_test_logi, y_pred_logi))
# print("Precission:",metrics.precision_score(y_test_logi, y_pred_logi))
# print("f1:",metrics.f1_score(y_test_logi, y_pred_logi))
# print("confusion_matrix:",'\n',metrics. confusion_matrix(y_test_logi, y_pred_logi))

"""

---

# Simple Neuronal Network"""

# import numpy as np
# from sklearn.model_selection import train_test_split
# import tensorflow as tf

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (8,)),
   #tf.keras.layers.Dense(4096, activation = 'relu'),
    tf.keras.layers.Dense(4096, activation = 'relu'),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0,2),
    tf.keras.layers.Dense(2, activation = 'softmax')
 ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy']
# )

# model.fit(X_train_MLP, y_train_MLP, epochs=15)






# history = model.fit(X_train_MLP,
#           y_train_MLP,
#           epochs=3,
#           validation_split=0.2)

# model.fit(X_test_MLP, y_test_MLP, epochs=15)
# from sklearn import metrics
# # Evaluate the model on the test data
# accuracyMLP = model.evaluate(X_test_MLP, y_test_MLP)
# print("Test Accuracy:", accuracyMLP)

# X_MLP = df2[['Species', 'Light_ISF',	'Soil',	'Sterile',	'Conspecific',	'Myco',	'SoilMyco',	'PlantDate']] #train
# y_MLP = df2["Event"] #test
# X_train_MLP, X_test_MLP, y_train_MLP, y_test_MLP = train_test_split(X_MLP, y_MLP, test_size=0.2, random_state=42)

# print(X_train_MLP.shape)
# print(X_test_MLP.shape)
# print(y_train_MLP.shape)
# print(y_test_MLP.shape)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape = (8,)),
#     #tf.keras.layers.Dense(4096, activation = 'relu'),
#     tf.keras.layers.Dense(4096, activation = 'relu'),
#     tf.keras.layers.Dense(1024, activation = 'relu'),
#     tf.keras.layers.Dense(1024, activation = 'relu'),
#     tf.keras.layers.Dense(256, activation = 'relu'),
#     tf.keras.layers.Dense(64, activation = 'relu'),
#     tf.keras.layers.Dropout(0,2),
#     tf.keras.layers.Dense(2, activation = 'softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy']
# )

# model.fit(X_train_MLP, y_train_MLP, epochs=15)

# history = model.fit(X_train_MLP,
#           y_train_MLP,
#           epochs=3,
#           validation_split=0.2)

# model.fit(X_test_MLP, y_test_MLP, epochs=15)

# from sklearn import metrics
# # Evaluate the model on the test data
# accuracyMLP = model.evaluate(X_test_MLP, y_test_MLP)
# print("Test Accuracy:", accuracyMLP)

"""

---



# LDA"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

dfLDA = df2.copy()

dfLDA = dfLDA.astype(float)

dfLDA['Event']=dfLDA["Event"].astype(int)

dfLDA = dfLDA.drop("Light_Cat", axis = 1)

cls = ["Species",	"Light_ISF",	"Soil",	"Sterile",	"Conspecific",	"Myco",	"SoilMyco",	"Event", "PlantDate"]

X = dfLDA.drop("Event", axis = 1)
y = dfLDA["Event"]

# Create a pair plot to visualize relationships between different features and species.
ax = sns.pairplot(dfLDA, hue='Event', markers=["X", "o"], hue_order = [1.0, 0.0])
plt.suptitle("Pair plot pentru dataset")
sns.move_legend(
    ax, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
plt.tight_layout()
plt.show()

# Visualizarea distributiilor cu ajutorul histogramelor
plt.figure(figsize=(14, 6))
for i, feature in enumerate(cls[:-1]):
    plt.subplot(4, 2, i + 1)
    sns.histplot(data = dfLDA, x = feature, hue = 'Event', hue_order = [1.0, 0.0], kde = True)
    plt.title(f'Distributia {feature}')

plt.tight_layout()
plt.show()

correlation_matrix = dfLDA.drop("Event", axis = 1).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Split the data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Apply Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components = 1) # Exista o singura componenta deoarece exista doar doua rezultate posibile la Event
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

"""## LDA SVM"""

from sklearn import svm
clf = svm.SVC(kernel='poly', degree=2, C=0.1, gamma=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#Assume 'y_test' and 'y_pred' are already defined
accuracy = accuracy_score(y_test, y_pred)
conf_m = confusion_matrix(y_test, y_pred)

#Display the accuracy
print(f'Accuracy: {accuracy:.2f}')

"""## LDA RF"""

classifier = RandomForestClassifier(max_depth = 5, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 2, n_estimators = 50, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# #Assume 'y_test' and 'y_pred' are already defined
accuracy = accuracy_score(y_test, y_pred)
conf_m = confusion_matrix(y_test, y_pred)

#Display the accuracy
print(f'Accuracy: {accuracy:.2f}')

#Display the confusion matrix as a heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(conf_m, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

"""## LDA SNN"""

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (8,)),
   #tf.keras.layers.Dense(4096, activation = 'relu'),
    tf.keras.layers.Dense(4096, activation = 'relu'),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0,2),
    tf.keras.layers.Dense(2, activation = 'softmax')
 ])

model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy']
 )

model.fit(X_train, y_train, epochs=15)






# history = model.fit(X_train_MLP,
#           y_train_MLP,
#           epochs=3,
#           validation_split=0.2)

# model.fit(X_test_MLP, y_test_MLP, epochs=15)
# from sklearn import metrics
# # Evaluate the model on the test data
# accuracyMLP = model.evaluate(X_test_MLP, y_test_MLP)
# print("Test Accuracy:", accuracyMLP)

history = model.fit(X_train,
          y_train,
          epochs=3,
          validation_split=0.2)

model.fit(X_test, y_test, epochs=15)
from sklearn import metrics
# Evaluate the model on the test data
accuracyMLP = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracyMLP)

"""# Trail and error"""

# from sklearn.model_selection import train_test_split
# X_MLP = df2[["Species",	"Light_ISF",	"Light_Cat", "Soil",	"Sterile",	"Conspecific",	"SoilMyco"]] #train
# y_MLP = df2["Event"] #test
# X_train_MLP, X_test_MLP, y_train_MLP, y_test_MLP = train_test_split(X_MLP, y_MLP, test_size=0.3)

# # import necessary libraries
# import numpy as np
# import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import InputLayer
# from tensorflow.keras.layers import Dense

# # Convert labels to one-hot encoded format
# y_train_MLP_one_hot = to_categorical(y_train_MLP, num_classes=4)
# y_test_MLP_one_hot = to_categorical(y_test_MLP, num_classes=4)
# X_train_MLP_one_hot = to_categorical(y_train_MLP, num_classes=4)
# X_test_MLP_one_hot = to_categorical(y_test_MLP, num_classes=4)
# np.random.seed(1337) # set a random seed
# model = Sequential()
# model.add(InputLayer(input_shape=(1947, 7)))
# #model = Sequential() # create a sequential model
# #model.add(InputLayer(input_shape = X_train_MLP.shape)) # add an input layer
# # the nex 2 lines add two fully connected (dense) layers with rectified linear unit (ReLU) activation functions
# model.add(Dense(8, activation="relu", name="layer1"))
# model.add(Dense(8, activation="relu", name="layer2"))
# model.add(Dense(4, activation = "softmax", name="layer3")) # add an output layer with three neuronsand a softmax activation function

# model.summary() # print a summary

# print("Shapes after train-test split:")
# print("x_train_MPL shape:", X_train_MLP_one_hot.shape)
# print("x_test_MPL shape:", X_test_MLP_one_hot.shape)
# print("Y_train_MPL shape:", y_train_MLP_one_hot.shape)
# print("Y_test_MPL shape:", y_test_MLP_one_hot.shape)

# from tensorflow.keras.utils import to_categorical

# # Convert labels to one-hot encoded format
# #y_train_MLP_one_hot = to_categorical(y_train_MLP, num_classes=4)
# #y_test_MLP_one_hot = to_categorical(y_test_MLP, num_classes=4)



# # initializing recall and precision metrics
# recall = tf.keras.metrics.Recall()
# precision = tf.keras.metrics.Precision()

# # compile the model
# model.compile(
#     optimizer = "adam",
#     loss = "categorical_crossentropy",
#     metrics = ["accuracy", precision, recall])
# history = model.fit(X_train_MLP_one_hot, y_train_MLP_one_hot,
#                     epochs=100,
#                     batch_size=5,
#                     verbose=1,
#                     validation_data=(X_test_MLP_one_hot, y_test_MLP_one_hot))

# # train the model
# #history = model.fit(X_train_MLP, y_train_MLP,
#  #                   epochs = 100,
#   #                  batch_size=5,
#    #                 verbose=0,
#     #                validation_data=(X_test_MLP, y_test_MLP))

# # containing the training history of the neural network model
# history.history.keys()

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# # Function to create a simple neural network model
# def create_simple_nn(input_shape):
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=(input_shape,)),
#         Dense(32, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ])
#     return model

# # Function to plot training history
# def plot_history(history):
#     plt.figure(figsize=(10, 5))

#     # Plot training & validation accuracy values
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('Model accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend(['Train', 'Test'], loc='upper left')

#     # Plot training & validation F1 score values
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['f1_score'])
#     plt.plot(history.history['val_f1_score'])
#     plt.title('Model F1 score')
#     plt.xlabel('Epoch')
#     plt.ylabel('F1 score')
#     plt.legend(['Train', 'Test'], loc='upper left')

#     plt.tight_layout()
#     plt.show()

# # Assume 'df' is your DataFrame with features and target variable
# # Extract features (input) and target variable (output)
# from sklearn.model_selection import train_test_split
# X_SNN = df2[["Species",	"Light_ISF",	"Light_Cat", "Soil",	"Sterile",	"Conspecific",	"SoilMyco"]] #train
# y_SNN = df2["Event"] #test
# X_train_SNN, X_test_SNN, y_train_SNN, y_test_SNN = train_test_split(X_SNN, y_SNN, test_size=0.3)

# # Get the number of features (input shape)
# input_shape = X_train_SNN.shape[1]

# # Create the neural network model
# model = create_simple_nn(input_shape)

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Define a custom F1 score metric
# def f1_score_metric(y_true, y_pred):
#     y_pred_binary = tf.round(y_pred)
#     return f1_score(y_true, y_pred_binary)

# # Train the model
# history = model.fit(X_train_SNN, y_train_SNN, epochs=10, batch_size=32, validation_data=(X_test_SNN, y_test_SNN),
#                     callbacks=[tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)],
#                     verbose=1)

# # Evaluate the model on the test set
# loss, accuracy = model.evaluate(X_test_SNN, y_test_SNN, verbose=0)
# y_pred = model.predict(X_test_SNN)
# f1 = f1_score(y_test_SNN, np.round(y_pred))

# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)
# print("Test F1 Score:", f1)

# # Plot training history
# plot_history(history)

# import numpy as np

# def sigmoid(x):
#   return 1 / (1 + np.exp(-x))

# from sklearn.model_selection import train_test_split
# X_SNN = df2[["Species",	"Light_ISF",	"Light_Cat", "Soil",	"Sterile",	"Conspecific",	"SoilMyco"]] #train
# y_SNN = df2["Event"] #test
# X_train_SNN, X_test_SNN, y_train_SNN, y_test_SNN = train_test_split(X_SNN, y_SNN, test_size=0.3)

# num_inputs = len(X_SNN)
# hidden_layer_neurons = 7
# np.random.seed(4)
# w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1
# w1

# num_outputs = len(y_SNN)
# w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1
# w2

# # taken from> https://gist.github.com/craffel/2d727968c3aaebd10359
# def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
#     '''
#     Draw a neural network cartoon using matplotilb.

#     :usage:
#         >>> fig = plt.figure(figsize=(12, 12))
#         >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])

#     :parameters:
#         - ax : matplotlib.axes.AxesSubplot
#             The axes on which to plot the cartoon (get e.g. by plt.gca())
#         - left : float
#             The center of the leftmost node(s) will be placed here
#         - right : float
#             The center of the rightmost node(s) will be placed here
#         - bottom : float
#             The center of the bottommost node(s) will be placed here
#         - top : float
#             The center of the topmost node(s) will be placed here
#         - layer_sizes : list of int
#             List of layer sizes, including input and output dimensionality
#     '''
#     n_layers = len(layer_sizes)
#     v_spacing = (top - bottom)/float(max(layer_sizes))
#     h_spacing = (right - left)/float(len(layer_sizes) - 1)
#     # Nodes
#     for n, layer_size in enumerate(layer_sizes):
#         layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
#         for m in range(layer_size):
#             circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
#                                 color='w', ec='k', zorder=4)
#             ax.add_artist(circle)
#     # Edges
#     for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
#         layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
#         layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
#         for m in range(layer_size_a):
#             for o in range(layer_size_b):
#                 line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
#                                   [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
#                 ax.add_artist(line)

# fig = plt.figure(figsize=(12, 12))
# ax = fig.gca()
# ax.axis('off')
# draw_neural_net(ax, .1, .9, .1, .9, [4, 5, 3])

# _x = np.linspace( -5, 5, 50 )
# _y = 1 / ( 1 + np.exp( -_x ) )
# plt.plot( _x, _y )

# learning_rate = 0.2 # slowly update the network
# error = []
# for epoch in range(1000):
#     # activate the first layer using the input
#     #   matrix multiplication between the input and the layer 1 weights
#     #   result is fed into a sigmoid function
#     l1 = 1/(1 + np.exp(-(np.dot(X_SNN, w1))))
#     # activate the second layer using first layer as input
#     l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
#     # find the average errorof this batch
#     #   using absolute, can use use square as well
#     er = (abs(y_SNN - l2)).mean()
#     error.append(er)

#     # BACKPROPAGATION / learning!
#     # find contribution of error on each weight on the second layer
#     l2_delta = (y - l2)*(l2 * (1-l2))
#     # update each weight in the second layer slowly
#     w2 += l1.T.dot(l2_delta) * learning_rate

#     # find contribution of error on each weight on the second layer w.r.t the first layer
#     l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
#     # udpate weights in the first layer
#     w1 += X.T.dot(l1_delta) * learning_rate

# print('Error:', er)

"""# Grid Search CV pe Random Forest"""

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators':[50, 100, 150],
    'max_depth':[None, 5, 10],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf':[1, 2, 4],
    'max_features':[None, 'sqrt'],
}