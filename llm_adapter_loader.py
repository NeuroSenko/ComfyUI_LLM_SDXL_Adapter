import torch
from safetensors.torch import load_file
import logging
import gc
import os
from .utils import get_llm_adapters, get_llm_adapter_path
from .llm_to_sdxl_adapter import LLMToSDXLAdapter

logger = logging.getLogger(__name__)


class LLMAdapterLoader:
    """
    ComfyUI node that loads LLM to SDXL adapter
    """
    
    def __init__(self):
        self.adapter = None
        self.current_adapter_path = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter_name": (get_llm_adapters(), {
                    "default": get_llm_adapters()[0] if get_llm_adapters() else None
                }),
            },
            "optional": {
                "llm_dim": ("INT", {
                    "default": 1152,
                    "min": 512,
                    "max": 4096
                }),
                "sdxl_seq_dim": ("INT", {
                    "default": 2048,
                    "min": 1024,
                    "max": 4096
                }),
                "sdxl_pooled_dim": ("INT", {
                    "default": 1280,
                    "min": 512,
                    "max": 2048
                }),
                "target_seq_len": ("INT", {
                    "default": 308,
                    "min": 64,
                    "max": 1024
                }),
                "n_wide_blocks": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 8
                }),
                "n_narrow_blocks": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 8
                }),
                "num_heads": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 32
                }),
                "dropout": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01
                }),
                "device": (["auto", "cuda:0", "cuda:1", "cpu"], {
                    "default": "auto"
                }),
                "force_reload": ("BOOLEAN", {
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = ("LLM_ADAPTER", "STRING")
    RETURN_NAMES = ("llm_adapter", "info")
    FUNCTION = "load_adapter"
    CATEGORY = "llm_sdxl"
    
    def load_adapter(self, adapter_name, llm_dim=1152, sdxl_seq_dim=2048, sdxl_pooled_dim=1280, 
                    target_seq_len=308, n_wide_blocks=2, n_narrow_blocks=3, num_heads=16, dropout=0.1, device="auto", force_reload=False):
        """Load and initialize the LLM to SDXL adapter"""
        if device == "auto":
            device = self.device
        
        adapter_path = get_llm_adapter_path(adapter_name)
        
        try:
            # Check if we need to reload
            if force_reload or self.adapter is None or self.current_adapter_path != adapter_path:
                # Clear previous adapter
                if self.adapter is not None:
                    del self.adapter
                    gc.collect()
                    torch.cuda.empty_cache()
                
                logger.info(f"Loading LLM to SDXL adapter from {adapter_path}")
                
                # Initialize adapter with specified parameters
                self.adapter = LLMToSDXLAdapter(
                    llm_dim=llm_dim,
                    sdxl_seq_dim=sdxl_seq_dim,
                    sdxl_pooled_dim=sdxl_pooled_dim,
                    target_seq_len=target_seq_len,
                    n_wide_blocks=n_wide_blocks,
                    n_narrow_blocks=n_narrow_blocks,
                    num_heads=num_heads,
                    dropout=dropout
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
                logger.info("LLM to SDXL adapter loaded successfully")
            
            info = f"Adapter: {adapter_path}\nDevice: {device}\nLLM dim: {llm_dim}\nSDXL seq dim: {sdxl_seq_dim}\nTarget seq len: {target_seq_len}"
            
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