echo Start downloading the ParkingLot dataset

mkdir -p data/parkinglot
cd data/parkinglot

# 01.zip
gdown https://drive.google.com/uc?id=1-O_Mle53luhaEEFh5YWWkThy29jzwjf_
unzip -q 01.zip -d 01
rm 01.zip

# 02.zip
gdown https://drive.google.com/uc?id=19sjinWpx5AvxUtRkprIVdaBjod8gkoyW
unzip -q 02.zip -d 02
rm 02.zip

# # 03.zip
# gdown https://drive.google.com/uc?id=1KrSoxT2mARoEIR08her0VUmdvBOjm779
# unzip -q 03.zip -d 03
# rm 03.zip

# # 04.zip
# gdown https://drive.google.com/uc?id=1_aLHcMdjuNir47UINDffN2_S30xvn0Gc
# unzip 04.zip
# rm 04.zip

# # 05.zip
# gdown https://drive.google.com/uc?id=1Pc5hHR0hxbCtVGOXQxPIigpqXh2ASFmb
# unzip 05.zip
# rm 05.zip

# # 06.zip
# gdown https://drive.google.com/uc?id=1hm8fGQPAAmviuFmT1WZl73qEjAqLZ6xX
# unzip 06.zip
# rm 06.zip

cd ../..

echo Finished downloading the ParkingLot dataset