#!/bin/bash
#Script to test the performance of the gpu depending on the number of threads per block
input_dir="images/original/"
output_dir="images/cuda/"

# Vérifier que le dossier de sortie existe sinon le créer
if [ ! -d "$output_dir" ]; then
  mkdir "$output_dir"
fi

# Itérer sur les fichiers GIF du dossier d'entrée
for file in "$input_dir"*.gif
do
  # Extraire le nom de fichier sans l'extension
  filename=$(basename -- "$file")
  extension="${filename##*.}"
  filename="${filename%.*}"

  # Exécuter l'exécutable sobelf
  for threads in 16 32 64 128 256 512 
  do
  ./sobelf "$file" "$output_dir$filename.gif" 1 "-t $threads" "-f collecting_cuda.csv"
  done
done
