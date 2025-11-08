import torch
from transformers import T5GemmaEncoderModel, AutoTokenizer, T5GemmaConfig
from safetensors.torch import load_file
import gc
import logging
from .utils import get_text_encoders, get_text_encoder_path

logger = logging.getLogger("LLM-SDXL-Adapter")


class T5GEMMALoaderS:
    """
    ComfyUI node that loads T5Gemma model from a single safetensors file
    instead of from_pretrained. Tokenizer is loaded from HuggingFace.
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
                "model_name": (get_text_encoders(), {
                    "default": get_text_encoders()[0] if get_text_encoders() else None
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

    def T5Gemma2b_encoder_init(self):
        """Initialize T5Gemma 2B encoder model with detailed config"""
        config = {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_logit_softcapping": 50.0,
            "classifier_dropout_rate": 0.0,
            "dropout_rate": 0.0,
            "final_logit_softcapping": 30.0,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 2304,
            "initializer_range": 0.02,
            "intermediate_size": 9216,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ],
            "max_position_embeddings": 8192,
            "model_type": "t5_gemma_module",
            "num_attention_heads": 8,
            "num_hidden_layers": 26,
            "num_key_value_heads": 4,
            "query_pre_attn_scalar": 256,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "torch_dtype": "bfloat16",
            "vocab_size": 256000,
            "pad_token_id": 0,
            "eos_token_id": [1, 107]
        }
        config = T5GemmaConfig(encoder=config, is_encoder_decoder=False)
        model = T5GemmaEncoderModel(config).to("cpu")
        return model

    def load_model(self, model_name, device="auto", force_reload=False):
        """Load T5Gemma Model from safetensors file and tokenizer from HuggingFace"""
        if device == "auto":
            device = self.device

        try:
            model_path = get_text_encoder_path(model_name)

            # Check if we need to reload
            if force_reload or self.model is None or self.current_model_path != model_path:
                # Clear previous model
                if self.model is not None:
                    del self.model
                    del self.tokenizer
                    gc.collect()
                    torch.cuda.empty_cache()

                logger.info(f"Loading T5Gemma Model from {model_path}")

                # Initialize model with detailed config
                self.model = self.T5Gemma2b_encoder_init().to(device)

                # Load state dict from safetensors
                self.model.load_state_dict(load_file(model_path))

                logger.info("T5Gemma Model weights loaded from safetensors")

                # Load tokenizer from HuggingFace
                t5_tokenizer_path = "Minthy/t5gemma-2b-2b-ul2-encoder-only"

                logger.info(f"Loading tokenizer from {t5_tokenizer_path}")

                self.tokenizer = AutoTokenizer.from_pretrained(t5_tokenizer_path)

                self.current_model_path = model_path
                logger.info("T5Gemma Model and tokenizer loaded successfully")

            info = f"Model: {model_path}\nDevice: {device}\nLoaded: {self.model is not None}"

            return (self.model, self.tokenizer, info)

        except Exception as e:
            logger.error(f"Failed to load T5Gemma Model: {str(e)}")
            raise Exception(f"Model loading failed: {str(e)}")



# Node mapping for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "T5GEMMALoaderS": T5GEMMALoaderS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T5GEMMALoaderS": "T5Gemma Loader (Safetensors)",
}
