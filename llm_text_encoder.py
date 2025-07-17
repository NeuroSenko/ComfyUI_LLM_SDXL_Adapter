import torch
import logging

logger = logging.getLogger(__name__)


class LLMTextEncoder:
    """
    ComfyUI node that encodes text using a loaded Language Model
    Supports various LLM architectures with chat templates
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLM_MODEL",),
                "tokenizer": ("LLM_TOKENIZER",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "masterpiece, best quality, 1girl, anime style"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are expert in understanding of user prompts for image generations. Create an image according to the prompt from user."
                }),
                "skip_first": ("INT", {
                    "default": 27,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("LLM_HIDDEN_STATES", "STRING")
    RETURN_NAMES = ("hidden_states", "info")
    FUNCTION = "encode_text"
    CATEGORY = "llm_sdxl"
    
    def encode_text(self, model, tokenizer, text, system_prompt="You are expert in understanding of user prompts for image generations. Create an image according to the prompt from user.", skip_first=27):
        """
        Encode text using Language Model and return hidden states
        """
        try:
            # Get model device
            device = next(model.parameters()).device
            
            # Prepare chat template
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt}
                    ]
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": text}
                    ]
                }
            ]
            
            # Apply chat template
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(device)
            
            # Generate hidden states
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Extract hidden states, skipping first tokens
            hidden_states = outputs['hidden_states'][-1][:, skip_first:, :].to(torch.float)
            
            # Prepare info
            info = f"Text: {text[:50]}...\nTokens after skip: {hidden_states.shape[1]}\nShape: {hidden_states.shape}"
            
            logger.info(f"Encoded text with shape: {hidden_states.shape}")
            
            return (hidden_states, info)
            
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise Exception(f"Text encoding failed: {str(e)}")


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "LLMTextEncoder": LLMTextEncoder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMTextEncoder": "LLM Text Encoder"
} 