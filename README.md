## Scalable GAN and C-GAN Implementation
- Trained on the MNIST dataset
- Binary Cross Entropy Loss was employed for simplicity. Future work can explore using Hinge loss to produce sharper images, or Wassertein loss for better training stability
## Results

![generated_images_grid2](https://github.com/PascualMeritaTorres/Scalable-CGAN---GAN/assets/91559051/faf92843-00c7-4bc3-ac64-9ad9a8bd490d)

## Run the Project
- Run the vanilla GAN
```sh
bash run_gan.sh
```
- Run the conditional GAN
```sh
bash run_cgan.sh
```
