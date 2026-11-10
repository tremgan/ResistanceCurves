# train

Trains the YOLO barbell detector and exports versioned ONNX + JSON sidecar to `models/`.

Needs `.env` with `ROBOFLOW_API_KEY` and `HF_TOKEN`.

```sh
# local
uv run main.py

# docker (trains, then syncs models/ to hf://buckets/tremgan/plate-weights)
docker build -f Dockerfile.train -t train .
docker run --env-file .env train
```
