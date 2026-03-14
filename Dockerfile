FROM nvcr.io/nvidia/tritonserver:23.10-py3

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY model_repository /models