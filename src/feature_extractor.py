"""
feature_extractor.py
Multimodal feature extractor using a single SigLIP checkpoint for both vision and text
(shared, aligned 768-d space) for SemEval-2027 Task 9 Subtask 2.
Extracts pooled visual representations for global image and K localized crops,
along with full-sequence token representations for accompanying text.
"""

from typing import Dict, List, Optional
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


class FeatureExtractor:
    """
    Multimodal feature extractor extracting:
      1. SigLIP vision tokens: (1+K, 768) float32 pooled representations
         (1 global image + K salient crops).
      2. SigLIP text tokens: (64, 768) float32 last_hidden_state representations
         and (64,) int64 mask (1 = real token, 0 = padding).
    """

    def __init__(
        self,
        vision_model_name: str = "google/siglip-base-patch16-224",
        text_model_name: str = "google/siglip-base-patch16-224",
        max_text_len: int = 64,
        device: Optional[str] = None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name
        self.max_text_len = max_text_len

        # 1. Vision processor & model
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        base_vision = AutoModel.from_pretrained(vision_model_name)
        if hasattr(base_vision, "vision_model"):
            self.vision_model = base_vision.vision_model
        else:
            self.vision_model = base_vision

        self.vision_model.to(self.device)
        self.vision_model.eval()
        for p in self.vision_model.parameters():
            p.requires_grad = False

        # 2. Text tokenizer & model (reuse the SigLIP text tower when the checkpoint is shared)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        if text_model_name == vision_model_name and hasattr(base_vision, "text_model"):
            self.text_model = base_vision.text_model
        else:
            base_text = AutoModel.from_pretrained(text_model_name)
            self.text_model = base_text.text_model if hasattr(base_text, "text_model") else base_text
        self.text_model.to(self.device)
        self.text_model.eval()
        for p in self.text_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_sample(
        self,
        text: Optional[str],
        full_image: Image.Image,
        crop_images: List[Image.Image],
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts pooled visual representations for (full_image + crop_images) and
        token representations for text.

        Returns:
            Dict containing:
                - "visual_tokens": (1+K, 768) torch.float32
                - "text_tokens": (max_text_len, 768) torch.float32
                - "text_mask": (max_text_len,) torch.int64
        """
        # Preprocess images
        all_imgs = [full_image.convert("RGB")]
        for crop in crop_images:
            all_imgs.append(crop.convert("RGB"))

        v_inputs = self.image_processor(images=all_imgs, return_tensors="pt")
        v_inputs = {k: v.to(self.device) for k, v in v_inputs.items()}

        v_out = self.vision_model(**v_inputs)
        # pooler_output has shape (1+K, 768)
        if hasattr(v_out, "pooler_output") and v_out.pooler_output is not None:
            visual_tokens = v_out.pooler_output
        else:
            visual_tokens = v_out.last_hidden_state.mean(dim=1)

        visual_tokens = visual_tokens.to(dtype=torch.float32)

        # Preprocess text
        if text is None:
            text = ""
        # SigLIP was trained on lowercased text padded to max_length.
        t_inputs = self.tokenizer(
            text.lower(),
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = t_inputs["input_ids"].to(self.device)
        # SigLIP's pad token equals its EOS token, so (input_ids != pad_id) would drop the EOS (and
        # everything for empty text). Count real tokens from an unpadded encoding instead.
        n_real = len(self.tokenizer(text.lower(), max_length=self.max_text_len, truncation=True)["input_ids"])
        text_mask = torch.zeros(self.max_text_len, dtype=torch.int64, device=self.device)
        text_mask[: max(1, n_real)] = 1

        t_out = self.text_model(input_ids=input_ids)
        # last_hidden_state.squeeze(0) has shape (max_text_len, 768)
        text_tokens = t_out.last_hidden_state.squeeze(0).to(dtype=torch.float32)

        return {
            "visual_tokens": visual_tokens,
            "text_tokens": text_tokens,
            "text_mask": text_mask,
        }


_SHARED_FEATURE_EXTRACTOR: Optional[FeatureExtractor] = None


def get_shared_feature_extractor(
    vision_model_name: str = "google/siglip-base-patch16-224",
    text_model_name: str = "google/siglip-base-patch16-224",
    max_text_len: int = 64,
    device: Optional[str] = None,
) -> FeatureExtractor:
    """Returns or creates a shared singleton FeatureExtractor instance."""
    global _SHARED_FEATURE_EXTRACTOR
    if _SHARED_FEATURE_EXTRACTOR is None:
        _SHARED_FEATURE_EXTRACTOR = FeatureExtractor(
            vision_model_name=vision_model_name,
            text_model_name=text_model_name,
            max_text_len=max_text_len,
            device=device,
        )
    return _SHARED_FEATURE_EXTRACTOR
