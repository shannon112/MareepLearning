# hw11_gan_img_generation: GAN on Anime Girl Images Generation
Dataset from https://crypko.ai/# , size = (71314, 96, 96, 3)  
<img src="https://github.com/shannon112/MareepLearning/blob/master/hw11_gan_img_generation/img/dataset.jpg" width=640/>

## DCGAN network architecture & loss
<img src="https://github.com/shannon112/MareepLearning/blob/master/hw11_gan_img_generation/img/dcgan_network.jpg" width=640/>

## Result of DCGAN
<img src="https://github.com/shannon112/MareepLearning/blob/master/hw11_gan_img_generation/img/p1.jpg" width=640/>

## WGAN_GP network architecture & loss
<img src="https://github.com/shannon112/MareepLearning/blob/master/hw11_gan_img_generation/img/wgan_gp_network.jpg" width=640/>

## Result of WGAN_GP
<img src="https://github.com/shannon112/MareepLearning/blob/master/hw11_gan_img_generation/img/p2.jpg" width=640/>

## Resource
homework video: https://www.youtube.com/watch?v=ByguarFA8GU&feature=youtu.be  
library limitation: https://reurl.cc/GVkjWD   

## Usage
```
bash hw11_p1.sh <checkpoint> <out_image>
bash hw11_p2.sh <checkpoint> <out_image>
bash train_p1.sh <face_dir> <checkpoint>
bash train_p2.sh <face_dir> <checkpoint>
```

## Homework version
https://github.com/NTU-speech-lab/hw11-shannon112
