from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from contextlib import asynccontextmanager
import numpy as np
import pickle as pk
import base64
import io

#data classes (request/response)
class ImageRequest(BaseModel):
    image: str

class PredictionResponse(BaseModel):
    prediction: float

# lifecicle - app
@asynccontextmanager
async def lifespan(app: FastAPI):
    global xgb_model
    with open('modelos/model_tree.pkl', 'rb') as f:
        xgb_model = pk.load(f)

    yield; del xgb_model

app = FastAPI(lifespan=lifespan)


#predict endpoint
@app.post('/predict', response_model=PredictionResponse)
async def predict(request: ImageRequest):
    img_bytes = base64.b64decode(request.image)
    img = Image.open(io.BytesIO(img_bytes)).resize((28, 28))
    img_array = np.array(img)
    img_array_gs = np.dot(img_array[..., :3], [.2989, .5870, .1140]).reshape(1, -1)

    return {'prediction': xgb_model.predict(img_array_gs)}


#healthcheck endpoint
@app.get('/healthcheck')
async def healthcheck():
    return {'status': 'alive'}