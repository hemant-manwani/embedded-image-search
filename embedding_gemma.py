"""
EmbeddingGemma 2: Local Multimodal (Text & Image) Embedding Engine
===================================================================
Uses Google's EmbeddingGemma 2 (quantized ONNX model) to project
both text and images into a shared 768-dimensional semantic space.
"""

import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
from tokenizers import Tokenizer

MODEL_REPO = "onnx-community/embeddinggemma-2-ONNX"
IMAGE_TOKEN_ID = 258880  # Special token ID for <|image|> in EmbeddingGemma2


class EmbeddingGemma2:
    def __init__(self, model_repo: str = MODEL_REPO):
        """Initialize the ONNX Runtime sessions for Vision and Text/Joint models."""
        print("Loading EmbeddingGemma 2 ONNX models...")
        
        # 1. Download/locate ONNX model files
        v_path = hf_hub_download(model_repo, "onnx/vision_encoder_q4.onnx")
        t_path = hf_hub_download(model_repo, "onnx/model_q4.onnx")
        tok_path = hf_hub_download(model_repo, "tokenizer.json")
        
        # 2. Configure ONNX inference sessions (optimized for CPU)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session_vision = ort.InferenceSession(v_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.session_text = ort.InferenceSession(t_path, sess_options=opts, providers=["CPUExecutionProvider"])
        
        # 3. Tokenizer and Image Preprocessor
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.processor = Gemma4ImageProcessorPil.from_pretrained(model_repo)
        
        # Empty placeholder tensors for unused modalities
        self._empty_aud = np.zeros((0, 512), dtype=np.float32)
        self._empty_vid = np.zeros((0, 512), dtype=np.float32)
        self._empty_img = np.zeros((0, 512), dtype=np.float32)
        
        print("EmbeddingGemma 2 ready! (Embedding dimension: 768)")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text query into a normalized 768-dimensional vector.
        
        Args:
            text: Input string query (e.g. 'a dog catching a frisbee')
        Returns:
            np.ndarray of shape (768,) with L2 norm = 1.0
        """
        enc = self.tokenizer.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        
        outputs = self.session_text.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_features": self._empty_img,
            "video_features": self._empty_vid,
            "audio_features": self._empty_aud
        })
        
        # outputs[1] is the pooled sentence_embedding: shape (1, 768)
        embedding = outputs[1][0]
        # Normalize just to be certain
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Embed a PIL Image into the same 768-dimensional vector space.
        
        Args:
            image: PIL Image object
        Returns:
            np.ndarray of shape (768,) with L2 norm = 1.0
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Step 1: Preprocess image into 16x16 patches and spatial position IDs
        inputs = self.processor(image, return_tensors="np")
        
        # Step 2: Extract vision features via Vision Encoder (outputs shape: (256, 512))
        img_features = self.session_vision.run(None, {
            "pixel_values": inputs["pixel_values"],
            "pixel_position_ids": inputs["image_position_ids"]
        })[0]
        
        num_tokens = img_features.shape[0]
        
        # Step 3: Pass image features through the Gemma multimodal transformer
        input_ids = np.full((1, num_tokens), IMAGE_TOKEN_ID, dtype=np.int64)
        attention_mask = np.ones((1, num_tokens), dtype=np.int64)
        
        outputs = self.session_text.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_features": img_features,
            "video_features": self._empty_vid,
            "audio_features": self._empty_aud
        })
        
        embedding = outputs[1][0]
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two unit vectors (dot product)."""
        return float(np.dot(vec_a, vec_b))
