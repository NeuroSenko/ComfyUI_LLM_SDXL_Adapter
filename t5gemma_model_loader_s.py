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
        config = {'encoder': {'return_dict': True,
                'output_hidden_states': False,
                'torchscript': False,
                'torch_dtype': 'bfloat16',
                'use_bfloat16': False,
                'pruned_heads': {},
                'tie_word_embeddings': True,
                'chunk_size_feed_forward': 0,
                'is_encoder_decoder': False,
                'is_decoder': False,
                'cross_attention_hidden_size': None,
                'add_cross_attention': False,
                'tie_encoder_decoder': False,
                'encoder_no_repeat_ngram_size': 0,
                'output_scores': False,
                'return_dict_in_generate': False,
                'forced_bos_token_id': None,
                'forced_eos_token_id': None,
                'remove_invalid_values': False,
                'exponential_decay_length_penalty': None,
                'suppress_tokens': None,
                'begin_suppress_tokens': None,
                'architectures': None,
                'finetuning_task': None,
                'id2label': {0: 'LABEL_0', 1: 'LABEL_1'},
                'label2id': {'LABEL_0': 0, 'LABEL_1': 1},
                'tokenizer_class': None,
                'prefix': None,
                'bos_token_id': 2,
                'pad_token_id': 0,
                'eos_token_id': 1,
                'sep_token_id': None,
                'task_specific_params': None,
                'problem_type': None,
                '_name_or_path': '',
                'classifier_dropout_rate': 0.0,
                'dropout_rate': 0.0,
                'model_type': 't5_gemma_module',
                'vocab_size': 256000,
                'max_position_embeddings': 8192,
                'hidden_size': 2304,
                'intermediate_size': 9216,
                'num_hidden_layers': 26,
                'num_attention_heads': 8,
                'head_dim': 256,
                'num_key_value_heads': 4,
                'initializer_range': 0.02,
                'rms_norm_eps': 1e-06,
                'use_cache': True,
                'rope_theta': 10000.0,
                'attention_bias': False,
                'attention_dropout': 0.0,
                'hidden_activation': 'gelu_pytorch_tanh',
                'query_pre_attn_scalar': 256,
                'sliding_window': 4096,
                'final_logit_softcapping': 30.0,
                'attn_logit_softcapping': 50.0,
                'layer_types': [
                'sliding_attention','full_attention',
                'sliding_attention','full_attention',
                'sliding_attention','full_attention',
                'sliding_attention','full_attention',
                'sliding_attention','full_attention',
                'sliding_attention','full_attention',
                'sliding_attention','full_attention',
                'sliding_attention','full_attention',
                'sliding_attention','full_attention',
                'sliding_attention','full_attention',
                'sliding_attention','full_attention',
                'sliding_attention','full_attention',
                'sliding_attention','full_attention'],
                'output_attentions': False},
                'return_dict': True,
                'output_hidden_states': False,
                'torchscript': False,
                'torch_dtype': 'bfloat16',
                'use_bfloat16': False,
                'tf_legacy_loss': False,
                'pruned_heads': {},
                'tie_word_embeddings': True,
                'chunk_size_feed_forward': 0,
                'is_encoder_decoder': False,
                'is_decoder': False,
                'cross_attention_hidden_size': None,
                'add_cross_attention': False,
                'tie_encoder_decoder': False,
                'no_repeat_ngram_size': 0,
                'encoder_no_repeat_ngram_size': 0,
                'bad_words_ids': None,
                'num_return_sequences': 1,
                'output_scores': False,
                'return_dict_in_generate': False,
                'forced_bos_token_id': None,
                'forced_eos_token_id': None,
                'remove_invalid_values': False,
                'exponential_decay_length_penalty': None,
                'suppress_tokens': None,
                'begin_suppress_tokens': None,
                'architectures': ['T5GemmaForConditionalGeneration'],
                'finetuning_task': None,
                'id2label': {0: 'LABEL_0', 1: 'LABEL_1'},
                'label2id': {'LABEL_0': 0, 'LABEL_1': 1},
                'tokenizer_class': None,
                'prefix': None,
                'bos_token_id': 2,
                'pad_token_id': 0,
                'eos_token_id': [1, 107],
                'sep_token_id': None,
                'decoder_start_token_id': None,
                'task_specific_params': None,
                'problem_type': None,
                'transformers_version': '4.54.0.dev0',
                'initializer_range': 0.02,
                'model_type': 't5gemma',
                'use_cache': True,
                'dropout_rate': 0.0,
                'attention_dropout': 0.0,
                'classifier_dropout_rate': 0.0,
                'output_attentions': False}
        config = T5GemmaConfig(encoder=config)
        
        config.is_encoder_decoder = False
        model = T5GemmaEncoderModel(config).to("cpu")
        model.config._attn_implementation = "sdpa"
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
                self.model = self.T5Gemma2b_encoder_init()

                # Load state dict from safetensors
                self.model.load_state_dict(load_file(model_path))
                self.model.post_init()
                self.model.tie_weights()
                self.model.eval()
                self.model.to(torch.bfloat16)
                self.model.to(device)

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
