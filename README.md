## Parallelization for Sobel Filter
### 0 Requirements
Vous devez être sur l'un des ordinatuers du réseau de l'Ecole Polytechnique pour que cela fonctionne, ou alors vous devez spécifié un chemin d'accès vers les bibliothèques : mpi, omp, cuda et cuda_runtime
Vous devez possédez le compilateur nvcc, mpicc
Si vous êtes sur l'un des ordinateur de l'X il vous faut ouvrir un terminal et renseigner la commande : source ./set_env.h à la racine du dossier sobelf pour obtenir les bibliothèques.
Nous aurions aimez mieux structurer le code mais cela génère trop de bug au niveau des scripts, nous preferons donc le laisser tel quel.
### 1 Structure 
Le dossier sobelf est divisé en 5 sous dossiers :
- Images : contient les gifs qui seront testés et filtrés, pour rajouter un filtre il faut le rajouter dans le dossier images/original
- include : contient les headers necessaires a la compilation du code, nous avosn rajoutés deux headers kernels.h et structs.h
- src : contient les fichiers sources qui seront compilés par le makefile 
- older_version : contient d'anciennes version du main qui permettent de retrouver les résultats présentés dans le rapport 
- obj : contient les .o qui seront linkés dans le makefile
### 2 Scripts provided 
Nous avons prevu plusieurs scripts mettant en place plusieurs scénario.
#### 2.1 Script .sh 
Les scripts .sh ne mettent pas en jeu MPI, ils sont au nombre de 3 : 
- sobelf_cuda_perf.sh : teste tout le jeu de données avec un nombre de threads différents à chaque fois permet de voir l'impact du nombre de threads par bloc sur la performance , stocke les resultats
- sobelf_cuda_against_vanilla.sh : lance le filtrage sur chaque image deux fois : 1 fois avec le gpu 1 fois sans, stocke les résultats dans collecting_cuda_vanilla.csv
#### 2.2 Script .batch
Ces scripts mettent en jeu MPI 
- Sobelf_small.batch : test unitaire qui permet de transformer un fichier, en choisissant les paramètres 
- Sobelf_full_test_cuda_MPI.batch test l'interêt de l'utilisation du gpu et de MPI à la fois : on va lancer pour chauqe image 16 filtering 8 sans gpu ou le nombre de processeurs varie de 1 à 8 et 8 avec le gpu toujours en gardant la variation du nombre de processeurs, stocke les résultats dans collecting_cuda_MPI.csv
- Sobelf.batch : test l'impact de la distribution du calcul avec MPI sur la performance. Stocke les résultats dans collecting_MPI.csv
### 3 Main provided 
Le main.cu que nous fournissons compile avec le makefile. Néanmoins si vous souhaitez retrouver nos résultats précédents il est possible de choisir l'un des fichiers du dossier older_version. Néanmoins des changements sont à effectuer : dans le makefile, ligne 16 changer main.cu par le nom de votre fichier, de même ligne 27. De plus vous devez changer la variable CC en fonction de ce que vous voulez faire. 
En mettant cc = nvcc cela devrait fonctionner. 
### 4 Notebook provided 
Nous avons deux notebooks python qui étudient les données exportées : 
- analyzing.ipynb : Illustre les résultats que nous avions trouvé en utilisant MPi
- analyzing_cuda_perf.ipynb : Illustre les résultats de l'utilisation de la carte graphique mais également de la combinaison MPI/CUDA

N'hésitez pas à me dire si quelque chose ne fonctionne pas comme prévu.
