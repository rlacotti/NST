import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision.transforms.functional as F

class NSTDataset(Dataset):
    def __init__(self, input_dir: str, transform=None):
        self.input_dir = input_dir
        self.img_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith((".jpg", ".png"))]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image
    
class aspectRatio:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img):
        w, h = img.size

        if h > w:
            new_h = self.size
            new_w = int(self.size * w / h)
        else:
            new_w = self.size
            new_h = int(self.size * h / w)

        img = F.resize(img, (new_h, new_w))

        delta_w = self.size - new_w
        delta_h = self.size - new_h

        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)

        img = F.pad(img, padding, fill=0)

        return img

def test():
    contentDir = "content_images"
    style1Dir = "style1_images"
    style2Dir = "style2_images"

    transforms = v2.Compose([
        aspectRatio(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    contentDataset = NSTDataset(contentDir, transform=transforms)
    style1Dataset = NSTDataset(style1Dir, transform=transforms)
    style2Dataset = NSTDataset(style2Dir, transform=transforms)

    testC = contentDataset[0]
    testS1 = style1Dataset[0]
    testS2 = style2Dataset[0]
    print(f"Content image shape: {testC.shape}")
    print(f"Style1 image shape: {testS1.shape}")
    print(f"Style2 image shape: {testS2.shape}")

if __name__ == "__main__":
    test()
