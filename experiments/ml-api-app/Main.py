
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
import io
from torchvision import transforms

# (1) 반드시 여러분의 모델 구조를 아래처럼 정의해야 함!
# 예시) EfficientNet-b0, ResNet, CustomModel 등
from torchvision.models import efficientnet_b0

def get_model():
    model = efficientnet_b0(num_classes=4)  # 클래스 개수에 맞게 수정
    return model

# (2) 템플릿 경로는 상대경로가 안전 (ex: 현재 실행폴더 기준)
TEMPLATE_DIR = './'  # index.html/result.html이 현재 폴더에 있다면

app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATE_DIR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# (3) 모델 불러오기 (state_dict 기반)
ckpt = torch.load('final_best_model_v2.pth', map_location=DEVICE, weights_only=False)
model = get_model().to(DEVICE)
model.load_state_dict(ckpt['model'])    # 만약 ckpt['model']이 state_dict인 경우!
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
labels = ['양호', '경증', '중등도', '중증']

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = model(x).argmax(1).item()
    pred_label = labels[y]
    return templates.TemplateResponse("result.html", {"request": request, "prediction": pred_label})
