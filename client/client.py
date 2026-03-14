# client/client.py
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import torch
import torch.nn.functional as F

TRITON_URL = "localhost:8000"
MODEL_NAME = "toxicity_classifier"

def predict(texts):
    client = InferenceServerClient(url=TRITON_URL)
    # BYTES ввод: numpy array dtype=object
    np_texts = np.array(texts, dtype=object)
    inp = InferInput("INPUT_TEXT", [len(texts)], "BYTES")
    inp.set_data_from_numpy(np_texts)
    out = InferRequestedOutput("OUTPUT")
    response = client.infer(model_name=MODEL_NAME, inputs=[inp], outputs=[out])
    logits = response.as_numpy("OUTPUT")  # shape (batch, N)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    # вернуть метки: argmax
    labels = probs.argmax(axis=1)
    return [{"text": t, "probs": p.tolist(), "label": int(l)} for t,p,l in zip(texts, probs, labels)]

if __name__ == "__main__":
    sample = ["Привет, как дела?", "Ты идиот"]
    print(predict(sample))