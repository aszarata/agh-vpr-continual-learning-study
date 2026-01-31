from PIL import Image
from typing import List
from torch.utils.data import Dataset
import torchvision.transforms as T

class BaseImageDataset(Dataset):
    def __init__(self, root_dir: str,  img_size: int, image_paths: List[str]):
        super().__init__()
        self.root_dir = root_dir
        self.images_paths = image_paths
        self.image_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        raise NotImplementedError()
    
    def _open_and_transform_image(self, idx):
        image_path = self.root_dir + "/" + self.images_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)
        return image