wget -O SleepEEG.zip https://figshare.com/ndownloader/articles/19930178/versions/1
wget -O Epilepsy.zip https://figshare.com/ndownloader/articles/19930199/versions/2
wget -O UCI.zip https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip

mkdir -p datasets/SleepEEG
mkdir -p datasets/Epilepsy
mkdir -p datasets/UCI_original

unzip SleepEEG.zip -d datasets/SleepEEG/
unzip  Epilepsy.zip -d datasets/Epilepsy/
unzip UCI.zip -d datasets/UCI_original/

unzip "datasets/UCI_original/UCI HAR Dataset.zip" -d datasets/UCI_original/

rm {SleepEEG,Epilepsy,UCI}.zip

rm -r "datasets/UCI_original/__MACOSX/"
rm -r "datasets/UCI_original/UCI HAR Dataset.names"
rm -r "datasets/UCI_original/UCI HAR Dataset.zip"

python setupDatasets.py