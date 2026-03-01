import torch
from torch.utils.data import random_split, DataLoader
from src.models.convnext_unet import ConvNeXtUNet
from src.data.cvc_seg_dataset import CVCClinicSegDataset

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Load dataset
dataset = CVCClinicSegDataset("data/raw/CVC-ClinicDB/PNG")

train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size

_, val_dataset = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=4)

# Load model
model = ConvNeXtUNet(pretrained=True)
model.load_state_dict(torch.load("outputs/day10/convnext_unet_best.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Check prediction distribution
with torch.no_grad():
    imgs, masks = next(iter(val_loader))
    imgs = imgs.to(DEVICE)

    logits = model(imgs)
    preds = torch.sigmoid(logits)

    print("Prediction mean:", preds.mean().item())
    print("Prediction max:", preds.max().item())
    print("Mask mean:", masks.mean().item())