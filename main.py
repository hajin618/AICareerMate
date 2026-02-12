from fastapi import FastAPI, UploadFile
from rag.loader import load_pdf

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile):

    contents = await file.read()

    path = f"data/uploads/{file.filename}"

    with open(path, "wb") as f:
        f.write(contents)

    text = load_pdf(path)

    return {"status": "uploaded", "text_length": len(text)}
