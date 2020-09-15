# 360SP dataset
    URL=https://cgv.cs.nthu.edu.tw/360SP/360SP-data.zip
    ZIP_FILE=./data/360SP.zip
    mkdir -p ./data/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/
    rm $ZIP_FILE
