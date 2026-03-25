# model_repository/toxicity_classifier/1/model.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import triton_python_backend_utils as pb_utils
import torch.nn.functional as F

MODEL_ID = "s-nlp/russian_toxicity_classifier"

class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            raw = in_tensor.as_numpy()
            texts = []
            for x in raw:
                if isinstance(x, (bytes, bytearray)):
                    texts.append(x.decode("utf-8"))
                else:
                    texts.append(str(x))

            # токенизация
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                out = self.model(**enc)
                logits = out.logits.cpu().numpy().astype(np.float32)

            out_tensor = pb_utils.Tensor("OUTPUT", logits)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)
        return responses

    def finalize(self):
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass