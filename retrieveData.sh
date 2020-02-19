#!/bin/bash

if [ -d "inputs" ]; then
    read -p "Do you wish to overwrite inputs folder (CAUTION: ACTUAL FOLDER WILL BE DELETED)?" yn
    case $yn in
        [Yy]* ) rm -r inputs;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no."; exit;;
    esac
fi

if [ -d "groundtruths" ]; then
    rm -r groundtruths
fi

mkdir inputs
mkdir groundtruths

echo 'Retrieving Samson...'
wget http://www.escience.cn/system/file?fileId=68596 -O 'inputs/samson.zip'
wget http://www.escience.cn/system/file?fileId=69115 -O 'inputs/samson_gt.zip'
unzip -qq inputs/samson.zip
mv Data_Envi/* inputs/
rm -r Data_Envi

unzip -qq inputs/samson_gt.zip
mv GroundTruth/end3.mat groundtruths/samson_gt.mat
rm -r GroundTruth/ inputs/*.zip


python mat2npz.py
