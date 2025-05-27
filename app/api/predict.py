from fastapi import APIRouter, UploadFile, File
from app.inference import predict_image

router = APIRouter()

@router.post('/predict')
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    prediction = predict_image(image=contents)
    return {"prediction": prediction}
