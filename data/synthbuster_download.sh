# Script to download Synthbuster dataset.
# Synthbuster dataset can be used only for non-commercial purposes.
# It is obtained from the RAISE dataset available on the web page: http://mmlab.science.unitn.it/RAISE/
# To use this dataset, cite these two papers:
# [1] Quentin Bammey. 
#    "Synthbuster: Towards detection of diffusion model generated images." IEEE OJSP, 2023.
# [2] Duc-Tien Dang-Nguyen, Cecilia Pasquini, Valentina Conotter, and Giulia Boato.
#    "RAISE: A Raw Images Dataset for Digital Image Forensics." In ACM MMSys, page 219–224, 2015.
#

wget -c https://zenodo.org/records/10066460/files/synthbuster.zip?download=1 -O synthbuster.zip
wget -c https://www.grip.unina.it/download/prog/DMimageDetection/real_RAISE_1k.zip -O real_RAISE_1k.zip
md5sum -c synthbuster_checksums.md5 
echo "unzipping ..."
unzip -q -n synthbuster.zip
unzip -q -n -d synthbuster real_RAISE_1k.zip
echo "done."
