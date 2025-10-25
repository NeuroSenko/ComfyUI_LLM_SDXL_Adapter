import os
import logging
import folder_paths

logger = logging.getLogger("LLM-SDXL-Adapter")

def get_llm_dict():
    """
    Get the dictionary of LLM checkpoints.
    Keys are the names of the LLM checkpoints, values are the paths to the LLM checkpoints.
    """
    llm_dict = {}
    llm_paths = [] # 空のリストとして初期化

    # 1. カスタムノードなどで登録されたパスを追加
    if "llm" in folder_paths.folder_names_and_paths:
        paths, _ = folder_paths.folder_names_and_paths["llm"]
        llm_paths.extend(paths) # extendでリストに追加

    # 2. ComfyUIの標準モデルパスを追加
    default_llm_path = os.path.join(folder_paths.models_dir, "llm")
    if os.path.exists(default_llm_path):
        # 重複を避けるため、まだリストになければ追加
        if default_llm_path not in llm_paths:
            llm_paths.append(default_llm_path)
    
    # 3. 大文字の"LLM"フォルダも念のため確認
    default_LLM_path_upper = os.path.join(folder_paths.models_dir, "LLM")
    if os.path.exists(default_LLM_path_upper):
        if default_LLM_path_upper not in llm_paths:
            llm_paths.append(default_LLM_path_upper)

    # 収集したすべてのパスをループ処理
    for llm_path in llm_paths:
        if os.path.exists(llm_path):
            for item in os.listdir(llm_path):
                item_path = os.path.join(llm_path, item)
                if os.path.isdir(item_path):
                    # Check if it's a valid model directory (contains config.json or similar)
                    if any(f in os.listdir(item_path) for f in ['config.json', 'model.safetensors', 'pytorch_model.bin']):
                        llm_dict[item] = item_path
                elif item.endswith(('.safetensors', '.bin', '.pt')):
                    llm_dict[item] = item_path

    return llm_dict

def get_llm_gguf_dict():
    """
    Get the dictionary of GGUF files.
    Keys are the names of the LLM checkpoints, values are the paths to the LLM checkpoints.
    """
    llm_gguf_dict = {}
    if "llm" in folder_paths.folder_names_and_paths:
        llm_paths, _ = folder_paths.folder_names_and_paths["llm"]
    elif os.path.exists(os.path.join(folder_paths.models_dir, "llm")):
        llm_paths = [os.path.join(folder_paths.models_dir, "llm")]
    else:
        llm_paths = [os.path.join(folder_paths.models_dir, "LLM")]

    for llm_path in llm_paths:
        if os.path.exists(llm_path):
            for item in os.listdir(llm_path):
                item_path = os.path.join(llm_path, item)
                if os.path.isfile(item_path):
                    if item_path.lower().endswith('.gguf'):
                        llm_gguf_dict[item] = llm_path

    return llm_gguf_dict
    
def get_adapters_dict():
    """
    Get the dictionary of LLM adapters.
    Keys are the names of the LLM adapters, values are the paths to the LLM adapters.
    """
    adapters_dict = {}
    if "llm_adapters" in folder_paths.folder_names_and_paths:
        adapters_paths, _ = folder_paths.folder_names_and_paths["llm_adapters"]
    else:
        adapters_paths = [os.path.join(folder_paths.models_dir, "llm_adapters")]

    for adapters_path in adapters_paths:
        if os.path.exists(adapters_path):
            for item in os.listdir(adapters_path):
                if item.endswith('.safetensors'):
                    adapters_dict[item] = os.path.join(adapters_path, item)

    return adapters_dict

def get_llm_checkpoints():
    """
    Get the list of available LLM checkpoints.
    """
    return list(get_llm_dict().keys())

def get_llm_ggufs():
    """
    Get the list of available LLM checkpoints packed in GGUF.
    """
    return list(get_llm_gguf_dict().keys())

def get_llm_adapters():
    """
    Get the list of available LLM adapters.
    """
    return list(get_adapters_dict().keys())

def get_llm_checkpoint_path(model_name):
    """
    Get the path to a LLM checkpoint.
    """
    llm_dict = get_llm_dict()

    if model_name in llm_dict:
        return llm_dict[model_name]
    else:
        raise ValueError(f"Model {model_name} not found")

def get_llm_gguf_path(model_name):
    """
    Get the path to a LLM checkpoint.
    """
    llm_dict = get_llm_gguf_dict()

    if model_name in llm_dict:
        return llm_dict[model_name]
    else:
        raise ValueError(f"Model {model_name} not found")

def get_llm_adapter_path(adapter_name):
    """
    Get the path to an LLM adapter.
    """
    adapters_dict = get_adapters_dict()

    if adapter_name in adapters_dict:
        return adapters_dict[adapter_name]
    else:
        raise ValueError(f"Adapter {adapter_name} not found")
