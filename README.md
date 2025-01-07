# Neural Style Transfer

This project demonstrates how to perform Neural Style Transfer (NST) as well as, Multi-Neural Style Transfer (MNST) to an image using the pretrained VGG19 model from PyTorch.

## Requirements
* Python 3.11
* PyTorch
* torchvision
* matplotlib
* Pillow

## Project structure
* **content_images/** --> Directory containing the content images.
* **style1_images/** --> Directory containing the first style images.
* **style2_images/** --> Directory containing the second style images.
* **generated_images/** --> Directory where the generated images will be saved.
* **ImageProcure.py** --> Contains the script that utilizes the Pixabay.com API to grab images from the site.
* **Preprocessing.py** --> Contains dataset loading and preprocessing functions.
* **NST.py** --> Main script to perform Neural Style Transfer.
* **MNST.py** --> Main script to perform Multi-Neural Style Transfer.

## How it works
1. *Content Image* --> The image whose content will be preserved.
2. *Style Image(s)* --> The style images the contribute their artistic features to the final generated image.
3. *VGG19 Model* --> Pretrained network to extract content and style features.
