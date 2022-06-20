wget https://zenodo.org/record/6663662/files/yolov5s.pt?download=1 -O ./download/yolov5s.pt
wget https://zenodo.org/record/6663662/files/data_25.zip?download=1 -O ./download/data_25.zip
wget https://zenodo.org/record/6663662/files/dev.zip?download=1 -O ./download/dev.zip
wget https://zenodo.org/record/6663662/files/head_swin_bnneck.zip?download=1 -O ./download/head_swin_bnneck.zip

unzip ./download/head_swin_bnneck.zip -d ./download/head_swin_bnneck/
unzip ./download/dev.zip -d ./download/dev/
unzip ./download/data_25.zip -d ./download/data_25/
