from dotenv import load_dotenv
from roboflow import Roboflow
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

IMGSZ = 416  # must match what the browser feeds the model

def main():

    data_dir = Path(__file__).parent / "data"

    if not data_dir.exists():
        load_dotenv()

        rf = Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY'))
        project = rf.workspace("test-qxaec").project("barbell_detection")
        version = project.version(1)

        version.download("yolo26", location=str(data_dir))


    model = YOLO("yolo26n.pt")
    results = model.train(data='dataset.yaml', epochs=10, imgsz=IMGSZ, project=str(Path(__file__).parent / "runs"))

    # after train(), `model` holds the best weights; export a versioned ONNX + sidecar for the web app
    name = f"barbell-{datetime.now():%Y%m%d-%H%M}"
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    shutil.move(model.export(format="onnx", imgsz=IMGSZ), models_dir / f"{name}.onnx")
    (models_dir / f"{name}.json").write_text(json.dumps({
        "imgsz": IMGSZ,
        "names": model.names,
        "dataset": "test-qxaec/barbell_detection/1",
        "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)"),
        "created": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))

if __name__ == '__main__':
    main()