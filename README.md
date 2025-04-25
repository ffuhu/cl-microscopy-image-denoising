# Generalizable Denoising of Microscopy Images using Generative Adversarial Networks and Contrastive Learning
Repository for the paper "Generalizable Denoising of Microscopy Images using Generative Adversarial Networks and Contrastive Learning".

You can reproduce our results by first getting the data ready:

```
python prepare_datasets/create_dataset_for_pix2pix_Convallaria_uint16.py
python prepare_datasets/combine_A_and_B.py
```
This will create paired images for the pix2pix training.

Then, you can run the `train.py` file with the corresponding paths to the training data.
