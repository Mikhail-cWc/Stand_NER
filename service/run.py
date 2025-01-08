from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from ner_kernel import SpacyNERModel, HFNERModel, BaseNERModel, FlairNERModel
from ner_kernel import Entity
from classes import NERResponse, EntityResponse, NERRequest

app = FastAPI(
    title="NER Service",
    description="Сервис для Named Entity Recognition",
    version="1.0.0"
)

model_registry = {
    "spacy": SpacyNERModel(model_name="ru_core_news_sm"),
    "hf": HFNERModel(model_name="dslim/bert-base-NER"),
    "flair": FlairNERModel(model_name="ner-fast")
}

origins = ["http://localhost:3001", "http://127.0.0.1:3001"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/predict", response_model=NERResponse)
def predict_ner(req: NERRequest):
    if req.model_name not in model_registry:
        raise HTTPException(status_code=404, detail="Модель не найдена")

    model: BaseNERModel = model_registry[req.model_name]
    entities: List[Entity] = model.predict_entities(req.text)

    response_entities = [EntityResponse(**e.__dict__) for e in entities]

    return NERResponse(entities=response_entities)


@app.get("/")
def root():
    return {"message": "NER Stand is alive."}
