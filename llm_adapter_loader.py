import torch
from safetensors.torch import load_file
import logging
import gc
import os
from .utils import get_llm_adapters, get_llm_adapter_path
from .llm_to_sdxl_adapter import LLMToSDXLAdapter

logger = logging.getLogger("LLM-SDXL-Adapter")

class LLMAdapterLoader:
    """
    ComfyUI node that loads LLM to SDXL adapter
    """
    
    def __init__(self):
        self.adapter = None
        self.current_adapter_path = None
        self.current_adapter_type = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def INPUT_TYPES(cls):
        adapter_types = ["gemma", "t5gemma"]
        return {
            "required": {
                "adapter_name": (get_llm_adapters(), {
                    "default": get_llm_adapters()[0] if get_llm_adapters() else None
                }),
                "type": (adapter_types, {"default": "gemma"}),
            },
            "optional": {
                "device": (["auto", "cuda:0", "cuda:1", "cpu"], {"default": "auto"}),
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("LLM_ADAPTER", "STRING")
    RETURN_NAMES = ("llm_adapter", "info")
    FUNCTION = "load_adapter"
    CATEGORY = "llm_sdxl"
    
    def load_adapter(self, adapter_name, type, device="auto", force_reload=False):
        """Load and initialize the LLM to SDXL adapter"""
        if device == "auto":
            device = self.device
        
        adapter_path = get_llm_adapter_path(adapter_name)
        
        # Adapter configuration presets per type
        ADAPTER_PRESETS = {
            "gemma": {
                "llm_dim": 1152,
                "sdxl_seq_dim": 2048,
                "sdxl_pooled_dim": 1280,
                "target_seq_len": 308,
                "n_wide_blocks": 2,
                "n_narrow_blocks": 3,
                "num_heads": 16,
                "dropout": 0.1,
            },
            "t5gemma": {
                "llm_dim": 2304,
                "sdxl_seq_dim": 2048,
                "sdxl_pooled_dim": 1280,
                "target_seq_len": 308,
                "n_wide_blocks": 3,
                "n_narrow_blocks": 3,
                "num_heads": 16,
                "dropout": 0.05,
            },
        }
        
        if type not in ADAPTER_PRESETS:
            raise ValueError(f"Unknown adapter type: {type}")
        config = ADAPTER_PRESETS[type]
        
        try:
            # Check if we need to reload
            if force_reload or self.adapter is None or self.current_adapter_path != adapter_path or self.current_adapter_type != type:
                # Clear previous adapter
                if self.adapter is not None:
                    del self.adapter
                    gc.collect()
                    torch.cuda.empty_cache()
                
                logger.info(f"Loading LLM to SDXL adapter from {adapter_path}")
                
                # Initialize adapter with specified parameters
                self.adapter = LLMToSDXLAdapter(
                    llm_dim=config["llm_dim"],
                    sdxl_seq_dim=config["sdxl_seq_dim"],
                    sdxl_pooled_dim=config["sdxl_pooled_dim"],
                    target_seq_len=config["target_seq_len"],
                    n_wide_blocks=config["n_wide_blocks"],
                    n_narrow_blocks=config["n_narrow_blocks"],
                    num_heads=config["num_heads"],
                    dropout=config["dropout"],
                )
                
                # Load checkpoint if file exists
                if os.path.exists(adapter_path):
                    checkpoint = load_file(adapter_path)
                    self.adapter.load_state_dict(checkpoint)
                    logger.info(f"Loaded adapter weights from {adapter_path}")
                else:
                    logger.warning(f"Adapter file not found: {adapter_path}, using initialized weights")
                
                # Move to device
                self.adapter.to(device)
                self.adapter.eval()
                
                self.current_adapter_path = adapter_path
                self.current_adapter_type = type
                logger.info("LLM to SDXL adapter loaded successfully")
            
            info = (
                f"Adapter: {adapter_path}\n"
                f"Type: {type}\n"
                f"Device: {device}\n"
                f"LLM dim: {config['llm_dim']}\n"
                f"SDXL seq dim: {config['sdxl_seq_dim']}\n"
                f"Target seq len: {config['target_seq_len']}"
            )
            
            return (self.adapter, info)
            
        except Exception as e:
            logger.error(f"Failed to load adapter: {str(e)}")
            raise Exception(f"Adapter loading failed: {str(e)}")



# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LLMAdapterLoader": LLMAdapterLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMAdapterLoader": "LLM Adapter Loader",
} 