
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch, io
from torchvision import transforms

TEMPLATE_DIR = '/Users/sindongjun/AIOpsCICD'
app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# 구글 드라이브에 있는 templates 경로 지정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('/Users/sindongjun/AIOpsCICD/model.pkl',
                  map_location=DEVICE, weights_only=False)
model = ckpt['model'].to(DEVICE)
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
    # 이미지 파일 읽기
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        y = model(x).argmax(1).item()
    y = 1  # 예시 (실제로는 모델로 예측)
    pred_label = labels[y]
    return templates.TemplateResponse("result.html", {"request": request, "prediction": pred_label})
