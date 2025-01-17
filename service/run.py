import requests
import re

from .classes import NERRequest, NERResponse, EntityResponse
from ner_kernel import SpacyNERModel, HFNERModel, BaseNERModel, FlairNERModel, Entity
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from ner_kernel import SpacyNERModel, HFNERModel, BaseNERModel, FlairNERModel
from ner_kernel import Entity
from .classes import NERResponse, EntityResponse, NERRequest

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
    if req.framework not in model_registry:
        raise HTTPException(status_code=404, detail="Модель не найдена")

    model: BaseNERModel = model_registry[req.framework].change_model(eq.model_name)
    entities: List[Entity] = model.predict_entities(req.text)

    response_entities = [EntityResponse(**e.__dict__) for e in entities]

    return NERResponse(entities=response_entities)


@app.get("/")
def root():
    return {"message": "NER Stand is alive."}


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

URL_REGEX = re.compile(
    r"^https?://"                 # начинается с http:// или https://
    r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,6}"  # домен (прим. example.com)
    r"(?::\d+)?(?:/.*)?$"                # порт и прочие пути - опционально
)


def is_url(text: str) -> bool:
    return bool(URL_REGEX.match(text))


@app.post("/predict", response_model=NERResponse)
def predict_ner(req: NERRequest):
    if req.framework not in model_registry:
        raise HTTPException(status_code=404, detail="Модель не найдена")

    if is_url(req.text):
        try:
            response = requests.get(req.text)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=400, detail=f"Не удалось загрузить страницу по ссылке: {str(e)}")
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        extracted_text = " ".join(p.get_text(separator=" ") for p in paragraphs).strip()
    else:
        extracted_text = req.text

    model_name: str = req.model_name if req.model_name else model_registry[req.framework].model_name
    model: BaseNERModel = model_registry[req.framework]
    model.change_model(model_name)

    entities: List[Entity] = model.predict_entities(extracted_text)

    response_entities = [EntityResponse(**e.__dict__) for e in entities]
    return NERResponse(entities=response_entities)
