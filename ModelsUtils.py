import imp
import os
from dataclasses import dataclass
from numpy import less




@dataclass
class ModelPaths():
    onnx_file_path: str = None
    classes_file_path:str = None




def get_onnx_models_paths(dir)->ModelPaths: 
    paths = []
    for (root, dirs, files) in os.walk(dir):
        for file_path in files:
            file_path = root+"\\"+file_path
            files_extensions = (".onnx", ".json")
            if file_path.endswith(files_extensions):
                paths.append(file_path)
    # returning a tuple of json and onnx path in order
    return [ModelPaths(path[1], path[0]) for path in zip(paths[0::2], paths[1::2])]




def get_master_model_path(master_model_name = "Categorization.onnx")->ModelPaths:
    paths = get_onnx_models_paths("onnx-models")
    for path in paths:
        if path.onnx_file_path.endswith(master_model_name):
            return path


def get_specific_model_path(target, paths):
    for path in paths:
        if path.onnx_file_path.endswith(target + ".onnx"):
            return path

def filter_predicitions_dict(preds_dict, percentage = 0.1):
    for key, value in preds_dict.items():
        if less(value, percentage):
            preds_dict[key] = 0.0 


def process_results(results):
    results = results[0].tolist()
    results = dict(zip([*range(len(results))] , results))
    filter_predicitions_dict(results)
    results = sorted(results.items(), key=lambda item: item[1])
    # returns a list of tuples (prediciton index, probability)
    results = [vaild_result for vaild_result in results if vaild_result[1]]
    return results

def get_response_from_results(models_results, MASTER_MODEL, SUB_MODELS):
    result = dict()
    result["data"] = list()
    response_dict = dict()
    for category_index, disease_list in models_results.items():
        cat_name = MASTER_MODEL.get_class_at_index(category_index)
        response_dict["category"] = cat_name
        response_dict["predection"] = list()
        # disease_list[0] -->disease_index , disease_list[1] -->probability of disease , 
        for idx in range(0, len(disease_list), 2):
            disease_index = disease_list[0]
            probability = disease_list[1]
            disease_name = SUB_MODELS[category_index].get_class_at_index(disease_index)
            response_dict["predection"].append({"diseases" : disease_name, "probability": probability})
    result["data"].append(response_dict)
    return result
    # return dumps(result)