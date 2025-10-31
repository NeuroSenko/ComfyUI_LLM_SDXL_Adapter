import torch
import logging
import copy
import re
import folder_paths
import comfy.utils
import comfy.lora as lora 

logger = logging.getLogger("LLM-SDXL-Adapter")


# This is necessary since the lora trainer uses seperate q,k,v projections. 

def convert_lora_to_old_format(lora_dict, prefix="lora_te"):
    """
    Converts LoRA dictionary from the LoRA format (separate q, k, v projections)
    to the current format (combined in_proj_weight for nn.MultiheadAttention).
    """
    converted_lora = {}
    mha_layers = {}  

    # regex to find the base name and the projection type (q, k, v)
    pattern = re.compile(rf"({prefix}_.+(?:_attn|_attention))_(q|k|v)_proj\.lora_(up|down)\.weight")


    # Pass 1: group MultiheadAttention weights and copy everything else
    for key, weight in lora_dict.items():
        match = pattern.match(key)
        if match:
            base_key, proj_type, direction = match.groups()
            
            if base_key not in mha_layers:
                mha_layers[base_key] = {}
            if direction not in mha_layers[base_key]:
                mha_layers[base_key][direction] = {}
            mha_layers[base_key][direction][proj_type] = weight

        elif "alpha" in key and ("_q_proj" in key or "_k_proj" in key or "_v_proj" in key):
            # alpha values should be the same for q, k, v. We create one alpha for the combined layer.
            new_key_base = key.split(".alpha")[0]
            if "_q_proj" in new_key_base:
                new_key_base = new_key_base.replace("_q_proj", "_in_proj")
            elif "_k_proj" in new_key_base:
                new_key_base = new_key_base.replace("_k_proj", "_in_proj")
            else:
                new_key_base = new_key_base.replace("_v_proj", "_in_proj")
            
            # only add the alpha value once for each combined layer
            if new_key_base + ".alpha" not in converted_lora:
                 converted_lora[new_key_base + ".alpha"] = weight
        else:
            # copy non-MultiheadAttention keys directly
            converted_lora[key] = weight

    # Pass 2: process the grouped MultiheadAttention weights
    for base_key, directions in mha_layers.items():
        # concat the up weights 
        if 'up' in directions and all(p in directions['up'] for p in ['q', 'k', 'v']):
            weights = directions['up']
            q_w, k_w, v_w = weights['q'], weights['k'], weights['v']
            
            # concat along the output dimension for the up-projection
            combined_up_weight = torch.cat([q_w, k_w, v_w], dim=0)
            
            new_key = f"{base_key}_in_proj.lora_up.weight"
            converted_lora[new_key] = combined_up_weight
        
        # Process the down weights by taking just one (they should be identical)
        if 'down' in directions and all(p in directions['down'] for p in ['q', 'k', 'v']):
            weights = directions['down']
            
            # input projection is shared, so we just take one of the down weights (e.g., from 'q')
            combined_down_weight = weights['q']
            
            new_key = f"{base_key}_in_proj.lora_down.weight"
            converted_lora[new_key] = combined_down_weight

    logger.info(f"Successfully converted {len(mha_layers)} MultiheadAttention LoRA layers to old format.")
    return converted_lora

def model_lora_keys_adapter(adapter_model, prefix="lora_te"):
    """
    Generates dictionary mapping LoRA keys to the corresponding weight names
    in adapter model (which uses standard nn.MultiheadAttention).
    """
    key_map = {}
    
    for name, module in adapter_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            lora_key = f"{prefix}_{name.replace('.', '_')}"
            key_map[lora_key] = f"{name}.weight"
        elif isinstance(module, torch.nn.MultiheadAttention):
            # map to the combined 'in_proj_weight'
            lora_key = f"{prefix}_{name.replace('.', '_')}_in_proj"
            key_map[lora_key] = f"{name}.in_proj_weight"
            
    return key_map


class T5GEMMALORALOADER:
    """
    ComfyUI node that loads a LoRA model and applies it to both the unet
    and adapter.
    First converts the LoRA to be compatible with the current
    adapter format that uses nn.MultiheadAttention.
    """

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "llm_adapter": ("LLM_ADAPTER",),
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_adapter": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "LLM_ADAPTER")
    FUNCTION = "load_lora"
    CATEGORY = "llm_sdxl"

    def load_lora(self, model, llm_adapter, lora_name, strength_model, strength_adapter):
        if strength_model == 0 and strength_adapter == 0:
            return (model, llm_adapter)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        # prevents key not loaded warnings for text encoder keys.
        lora_dict_unet = {key: value for key, value in lora_dict.items() if "lora_unet" in key}
        
        model_patched = model.clone()
        key_map_unet = lora.model_lora_keys_unet(model_patched.model)
        
        # Use the filtered dictionary for unet
        patches_model = lora.load_lora(lora_dict_unet, key_map_unet)
        
        model_patched.add_patches(patches_model, strength_model)
        logger.info(f"Applied {len(patches_model)} LoRA patches to UNet from '{lora_name}' with strength {strength_model}")
        
        adapter_patched = copy.deepcopy(llm_adapter)
        
        # filter text encoder keys before the conversion
        lora_dict_adapter = {key: value for key, value in lora_dict.items() if "lora_te" in key}
        
        if not lora_dict_adapter:
            logger.warning(f"No text encoder keys ('lora_te') found in LoRA '{lora_name}'. Skipping adapter patching.")
            return (model_patched, adapter_patched)

        logger.info(f"Converting LoRA '{lora_name}' for compatibility with the current adapter format...")
        lora_dict_converted_adapter = convert_lora_to_old_format(lora_dict_adapter, "lora_te")
        key_map_adapter = model_lora_keys_adapter(adapter_patched, "lora_te")
        patches_adapter = lora.load_lora(lora_dict_converted_adapter, key_map_adapter, log_missing=True)
        
        if not patches_adapter:
            logger.warning(f"No matching keys found in LoRA '{lora_name}' for the T5 Adapter after conversion. Returning original adapter.")
        else:
            adapter_sd = adapter_patched.state_dict()
            original_weights = {} 
            patched_keys_count = 0

            for key, weight in adapter_patched.named_parameters():
                if key in patches_adapter:
                    patch_tuple = (strength_adapter, patches_adapter[key], 1.0, None, None)
                    new_weight = lora.calculate_weight([patch_tuple], weight, key, original_weights=original_weights)

                    adapter_sd[key] = new_weight
                    patched_keys_count += 1

            adapter_patched.load_state_dict(adapter_sd)
            logger.info(f"Successfully applied {patched_keys_count} LoRA patches from '{lora_name}' to T5 Adapter with strength {strength_adapter}")

        return (model_patched, adapter_patched)


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "T5GEMMALORALOADER": T5GEMMALORALOADER
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T5GEMMALORALOADER": "T5Gemma Load LoRA"
}