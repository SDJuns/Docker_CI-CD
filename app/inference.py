import torch
from torchvision import transforms
from PIL import Image
from app.model.model import get_model
from app.train.config import DEVICE, MODEL_WEIGHT_PATH

model = get_model().to(DEVICE)
model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE)['model'])
model.eval()

data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_image(image=None, image_path=None):
    image = Image.open(image_path).convert('RGB') if image_path else image
    image = data_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
    return output.argmax(dim=1).item()
