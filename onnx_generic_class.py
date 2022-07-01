import imp
import cv2
import onnxruntime as ort
import numpy as np
from PIL import Image
from dataclasses import dataclass
import json


@dataclass
class ModelPaths():
    onnx_file_path: str = None
    classes_file_path:str = None





class OnnxModelRunner:
    def __init__(self, model_path, classes_json_path):
        self._is_running = False
        self.paths = ModelPaths(model_path, classes_json_path)
    
        self.classes_dict = None
        self.load_model()
        self.load_classes_names()
     
    
    def load_model(self):
        # providers could be changed to CUDA if the server had a dedicated GPU
        providers = ['CPUExecutionProvider']
        self.inference_session = ort.InferenceSession(self.paths.onnx_file_path, providers=providers)
        self.input_name = self.inference_session.get_inputs()[0].name
        self.label_name = self.inference_session.get_outputs()[0].name
        self._is_running = True
         

    
    def load_json_file(self):
        with open(self.paths.classes_file_path) as json_file:
            return json.load(json_file)

    def load_classes_names(self, ):
        json_file = self.load_json_file()
        # made the key to be the integer values as they are the output of the models
        json_dict = {value:key for key, value in json_file.items()}
        self.classes_dict = json_dict

    def get_class_at_index(self, index):
        return self.classes_dict[index]
        
    async def get_img(self, img_file, img_size):
        with Image.open(img_file) as img:
            img = img.convert('RGB')
            img = np.array(img)
            img = cv2.resize(img, img_size, cv2.INTER_LINEAR)/225.0
            img = img.reshape(-1, img_size[0], img_size[1], 3)
            return img

    async def predict(self, img_path, img_size = (224, 224)):
        img = await self.get_img(img_path, img_size)
        predictions = self.inference_session.run([self.label_name], {self.input_name:img.astype(np.float32)})[0]
        return predictions

    def get_path(self):
        return self.model_path

