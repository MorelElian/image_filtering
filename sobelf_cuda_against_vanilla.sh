#!/bin/bash
#test cuda agianst the vanilla version
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
  for on_gpu in 0 1
    do
    ../sobelf "$file" "$output_dir$filename.gif" "1" "-g $on_gpu" "-f collecting_cuda_vanilla.csv"
    done
done
