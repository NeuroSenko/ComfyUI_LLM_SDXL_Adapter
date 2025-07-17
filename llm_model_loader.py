import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import logging
from .utils import get_llm_checkpoints, get_llm_checkpoint_path

logger = logging.getLogger(__name__)


class LLMModelLoader:
    """
    ComfyUI node that loads Language Model and tokenizer
    Supports various LLM architectures (Gemma, Llama, Mistral, etc.)
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_path = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (get_llm_checkpoints(), {
                    "default": get_llm_checkpoints()[0] if get_llm_checkpoints() else None
                }),
            },
            "optional": {
                "device": (["auto", "cuda:0", "cuda:1", "cpu"], {
                    "default": "auto"
                }),
                "force_reload": ("BOOLEAN", {
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = ("LLM_MODEL", "LLM_TOKENIZER", "STRING")
    RETURN_NAMES = ("model", "tokenizer", "info")
    FUNCTION = "load_model"
    CATEGORY = "llm_sdxl"
    
    def load_model(self, model_name, device="auto", force_reload=False):
        """Load Language Model and tokenizer"""
        if device == "auto":
            device = self.device
                
        try:
            model_path = get_llm_checkpoint_path(model_name)

            # Check if we need to reload
            if force_reload or self.model is None or self.current_model_path != model_path:
                # Clear previous model
                if self.model is not None:
                    del self.model
                    del self.tokenizer
                    gc.collect()
                    torch.cuda.empty_cache()
                
                logger.info(f"Loading Language Model from {model_path}")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    output_hidden_states=True,
                    trust_remote_code=True
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                self.current_model_path = model_path
                logger.info("Language Model loaded successfully")
            
            info = f"Model: {model_path}\nDevice: {device}\nLoaded: {self.model is not None}"
            
            return (self.model, self.tokenizer, info)
            
        except Exception as e:
            logger.error(f"Failed to load Language Model: {str(e)}")
            raise Exception(f"Model loading failed: {str(e)}")



# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LLMModelLoader": LLMModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMModelLoader": "LLM Model Loader",
} 