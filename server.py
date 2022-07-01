from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from OnnxModel import *
from ModelsUtils import *
import json
import uvicorn



app = FastAPI()






MASTER_MODEL_PATH = get_master_model_path("Categorization.onnx")
MASTER_MODEL = OnnxModelRunner(model_path=MASTER_MODEL_PATH.onnx_file_path, classes_json_path=MASTER_MODEL_PATH.classes_file_path)



from typing import List
SUB_MODELS: List[OnnxModelRunner] = list()

def load_models_in_order(paths,  order):
    models_order = order
    for key in models_order:
        path = get_specific_model_path(models_order[key], paths)
        if path:
            model = OnnxModelRunner(model_path=path.onnx_file_path, classes_json_path=path.classes_file_path)
            SUB_MODELS.append(model)
    


load_models_in_order(get_onnx_models_paths("onnx-models"), MASTER_MODEL.classes_dict)



async def run_models(image_path):
    categories = process_results(await MASTER_MODEL.predict(image_path))
    submodels = SUB_MODELS
    models_result = dict()
    for index, probabilty in categories:
        models_result[index] = process_results( await submodels[index].predict(image_path))[0]
    # the result is as follows (category_index: [diseases inside category index, its probability])
    return models_result


  
@app.post("/upload")
async def create_upload_file(img: UploadFile = File(...)):
    results = await run_models(img.file)
    json_response = get_response_from_results(results, MASTER_MODEL, SUB_MODELS)
    json_response = JSONResponse(content=json_response)
    return json_response

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")