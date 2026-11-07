from dotenv import load_dotenv
from roboflow import Roboflow
import os
from pathlib import Path
from ultralytics import YOLO


def main():

    data_dir = Path(__file__).parent / "data"

    if not data_dir.exists():
        load_dotenv()

        rf = Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY'))
        project = rf.workspace("test-qxaec").project("barbell_detection")
        version = project.version(1)

        version.download("yolo26", location=str(data_dir))


    model = YOLO("yolo26n.pt") 
    model.train(data='dataset.yaml', epochs=3, project=str(Path(__file__).parent / "runs"))

if __name__ == '__main__':
    main()