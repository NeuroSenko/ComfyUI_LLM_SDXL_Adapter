import torch
import logging

logger = logging.getLogger("LLM-SDXL-Adapter")


class T5GEMMATextEncoderV2:
    """
    Simplified ComfyUI node that combines text encoding and adapter application in one step.
    V2 version - all-in-one node with hardcoded parameters, only text field is editable.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "llm_tokenizer": ("LLM_TOKENIZER",),
                "llm_adapter": ("LLM_ADAPTER",),
                "text": ("STRING", {"multiline": True, "default": "masterpiece, best quality"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "llm_sdxl"

    def encode(self, llm_model, llm_tokenizer, llm_adapter, text):
        """
        Encode text using LLM and apply adapter to produce SDXL conditioning.
        All parameters are hardcoded for simplicity.
        """
        try:
            # Hardcoded parameters
            max_length = 512
            device = "cuda"
            width = 1024
            height = 1024
            target_width = 1024
            target_height = 1024
            crop_w = 0
            crop_h = 0

            # Tokenize
            inputs = llm_tokenizer(
                text + "<eos>",
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # Generate hidden states
            with torch.no_grad():
                outputs = llm_model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state.to(torch.float32)

                # Apply adapter
                prompt_embeds, pooled_output = llm_adapter(hidden_states, attention_mask=attention_mask)

            prompt_embeds = prompt_embeds.cpu().contiguous()
            pooled_output = pooled_output.cpu().contiguous()

            # Build conditioning metadata
            meta = {
                "pooled_output": pooled_output,
                "width": width,
                "height": height,
                "target_width": target_width,
                "target_height": target_height,
                "crop_w": crop_w,
                "crop_h": crop_h
            }

            conditioning = [[prompt_embeds, meta]]

            logger.info(f"Encoded text to conditioning: {prompt_embeds.shape}")

            return (conditioning,)

        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise Exception(f"Text encoding failed: {str(e)}")


# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "T5GEMMATextEncoderV2": T5GEMMATextEncoderV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T5GEMMATextEncoderV2": "T5Gemma Text Encode v2"
}
