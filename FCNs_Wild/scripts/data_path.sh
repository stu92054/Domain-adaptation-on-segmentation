# You need to Use this shell in ./Domain-adpatation-on-segmentation/FCNs_Wild
# sh data_path.sh <labelfile name> <Your dataset folder> 
# example: sh data_path.sh ./data/Taipei.txt /media/dataset/NMD/... util to images folder
cp $1 $1.backup
data_path=$2
data_path="\"$data_path\""
awk -F'/' '{print $data_path $11}' $1 > $1
