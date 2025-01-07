import os
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.models import vgg19, VGG19_Weights
from torchvision.transforms import v2, ToPILImage
from torch.utils.data import DataLoader
from Preprocessing import NSTDataset, aspectRatio
from torchvision.utils import save_image


# Load content and style images, apply transformations, return them as tensors
def load_data(content_dir: str, style1_dir: str, transform=None):
    if transform is None:
        transforms = v2.Compose([
                aspectRatio(224),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    content_dataset = NSTDataset(content_dir, transform=transforms)
    style1_dataset = NSTDataset(style1_dir, transform=transforms)

    content_loader = DataLoader(content_dataset, batch_size=len(content_dataset))
    style1_loader = DataLoader(style1_dataset, batch_size=len(style1_dataset))

    # Convert to an iterator, and return the next item from it (since no previous next calls, we just grab the first item)
    content_images = next(iter(content_loader))
    style1_images = next(iter(style1_loader))

    return content_images, style1_images

# Invert the transformations applied to the now existing tensors, and convert back to images
def unloader(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Clone the image and remove the batch dimension
    img = img.clone()
    img = img.squeeze(0)

    # To undo the normalization, we multiply by the std and add back the mean
    for i, m, s in zip(img, mean, std):
        i.mul_(s).add_(m)

    return ToPILImage()(img)

# Plot the original images
def imshow(img):
    img = unloader(img)
    plt.imshow(img)
    plt.axis('off')


# Grab the features from the pretrained model at the given layers, disregarding the FC layers
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # Represent conv1_1, conv2_1, etc. up until the final layer we are interested in
        self.feature_maps = ['0', '5', '10', '19', '28']
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:29]

    def forward(self, output):
        features = []
        for num, layer in enumerate(self.vgg):
            output = layer(output)

            if str(num) in self.feature_maps:
                features.append(output)

        return features



def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # Create output folder for generated image
    genDir = "generated_images"
    if not os.path.exists(genDir):
        os.makedirs(genDir)

    # Path to image dirs
    contentDir = "content_images"
    style1Dir = "style1_images"

    # Load and transform image sets
    content_images, style1_images = load_data(contentDir, style1Dir)
    
    # Select whatever image you want via index (requires unloading and imshow to decide)
    # Generated image can just be a clone of the content image, requires_grad_ to record optimization operations on tensor
    content_img = content_images[6].to(device)
    style1_img = style1_images[6].to(device)
    generated_img = content_img.clone().requires_grad_(True)


    # Hyperparams (Tune params on how you would like generated image to look)
    steps = 3500
    lr = 0.001
    alpha = 1
    beta = 5
    optimizer = Adam([generated_img], lr=lr)

    model = VGG().to(device).eval()

    for step in range(steps):
        # Pass through model to extract features at each layer
        generated_features = model(generated_img)
        content_features = model(content_img)
        style1_features = model(style1_img)

        # Initialize both style and content loss
        style1_loss = content_loss = 0

        for gen_feature, con_feature, sty_feature in zip(generated_features, content_features, style1_features):
            channel, height, width = gen_feature.shape
            # Content loss formula
            content_loss += torch.mean((gen_feature - con_feature)**2)

            # Gram matrix for generated image --> G = F * F^T
            G = gen_feature.view(channel, height*width).mm(
                gen_feature.view(channel, height*width).t()
            )

            # Gram matrix for style image --> A = S * S^T
            A = sty_feature.view(channel, height*width).mm(
                sty_feature.view(channel, height*width).t()
            )

            # Style loss Formula
            style1_loss += torch.mean((G - A)**2) / (channel * height * width)

        # Total loss Formula
        total_loss = alpha*content_loss + beta*style1_loss

        # Backprop and update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # At each interval, the image saved is overwitten iteratively to the same png file within the genDir folder
        if step % 500 == 0:
            print(f"Step {step}, Loss: {total_loss}")
            file_path = os.path.join(genDir, 'NSTsamurai.png')
            save_image(generated_img, file_path)

    '''
    # Plot original Content image
    plt.figure()
    imshow(content_img)

    # Plot original Style image
    plt.figure()
    imshow(style1_img)

    plt.show()
    '''

if __name__ =="__main__":
    main()
