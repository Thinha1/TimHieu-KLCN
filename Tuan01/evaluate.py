import os
import math

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("USE_TF", "0")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms, models
from transformers import CLIPProcessor, CLIPVisionModel, ViTModel, ViTImageProcessor
import re
from collections import Counter
from tqdm import tqdm
import argparse
import random

from pathlib import Path

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    corpus_bleu = SmoothingFunction = meteor_score = None


class CNNEncoder(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        if model_name == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            modules = list(resnet.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.out_dim = 2048
        elif model_name == "efficientnet_b3":
            effnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.model = effnet.features
            self.out_dim = 1536
        else:
            raise ValueError(f"Model {model_name} not supported by CNNEncoder")

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        features = self.model(images)  # [B, C, H, W]
        features = features.permute(0, 2, 3, 1)  # [B, H, W, C]
        features = features.view(features.size(0), -1, features.size(-1))  # [B, H*W, C]
        return features


class ViTEncoderModel(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.model = ViTModel.from_pretrained(model_name)
        self.out_dim = 768
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        outputs = self.model(pixel_values=images)
        return outputs.last_hidden_state


class CLIPEncoderModel(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.out_dim = 768
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        outputs = self.model(pixel_values=images)
        return outputs.last_hidden_state


class CustomTokenizer:
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, captions=None, min_freq=2):
        self.min_freq = min_freq
        self.word_to_id = {
            self.PAD_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3
        }
        self.id_to_word = {
            0: self.PAD_TOKEN,
            1: self.BOS_TOKEN,
            2: self.EOS_TOKEN,
            3: self.UNK_TOKEN
        }

        if captions is not None:
            self.build_vocab(captions)

    def tokenize(self, text):
        text = str(text).lower().strip()
        return re.findall(r"\b\w+\b", text)

    def build_vocab(self, captions):
        counter = Counter()

        for caption in captions:
            counter.update(self.tokenize(caption))

        next_id = len(self.word_to_id)
        for word, frequency in counter.items():
            if frequency >= self.min_freq:
                self.word_to_id[word] = next_id
                self.id_to_word[next_id] = word
                next_id += 1

        print(f"[*] Custom vocabulary size: {len(self.word_to_id)}")

    def encode(self, text, max_length):
        tokens = self.tokenize(text)
        token_ids = [self.bos_token_id]

        for token in tokens:
            token_ids.append(self.word_to_id.get(token, self.unk_token_id))

        token_ids.append(self.eos_token_id)

        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            token_ids[-1] = self.eos_token_id

        attention_mask = [1] * len(token_ids)
        padding_length = max_length - len(token_ids)
        token_ids.extend([self.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long)
        )

    def decode(self, token_ids):
        words = []

        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.eos_token_id:
                break
            if token_id in [self.pad_token_id, self.bos_token_id]:
                continue
            words.append(self.id_to_word.get(token_id, self.UNK_TOKEN))

        return " ".join(words)

    @property
    def vocab_size(self):
        return len(self.word_to_id)

    @property
    def pad_token_id(self):
        return self.word_to_id[self.PAD_TOKEN]

    @property
    def bos_token_id(self):
        return self.word_to_id[self.BOS_TOKEN]

    @property
    def eos_token_id(self):
        return self.word_to_id[self.EOS_TOKEN]

    @property
    def unk_token_id(self):
        return self.word_to_id[self.UNK_TOKEN]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class UniversalTransformerCaptioner(nn.Module):
    def __init__(self,
                 encoder_type="resnet50",
                 vocab_size=50257,
                 d_model=512,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()

        if encoder_type == "resnet50":
            self.encoder = CNNEncoder("resnet50")
        elif encoder_type == "efficientnet_b3":
            self.encoder = CNNEncoder("efficientnet_b3")
        elif encoder_type == "vit_b_16":
            self.encoder = ViTEncoderModel()
        elif encoder_type == "clip_vit_b_16":
            self.encoder = CLIPEncoderModel()
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        encoder_hidden_size = self.encoder.out_dim
        self.enc_projection = nn.Linear(encoder_hidden_size, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, pixel_values, input_ids, labels=None, attention_mask=None):
        with torch.no_grad():
            image_embeds = self.encoder(pixel_values)

        image_embeds = self.enc_projection(image_embeds)
        seq_len = input_ids.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(input_ids.device)

        tgt_emb = self.embedding(input_ids)
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)

        tgt_key_padding_mask = None
        if attention_mask is not None:
            tgt_key_padding_mask = (attention_mask == 0)

        output = self.decoder(
            tgt=tgt_emb,
            memory=image_embeds,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        logits = self.fc_out(output)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss, logits


def _unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def load_model(model, checkpoint_path, optimizer=None, scaler=None, device="cuda", strict=True):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        _unwrap_model(model).load_state_dict(checkpoint, strict=strict)
        return 0, float("inf")
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"[*] Đã load checkpoint: {checkpoint_path} | resume từ epoch {start_epoch + 1}")
    return start_epoch, best_val_loss


def get_checkpoint_vocab_size(checkpoint_path, device="cpu"):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    embedding_weight = state_dict.get("embedding.weight")
    if embedding_weight is None:
        return None
    return embedding_weight.shape[0]

class ImagePreprocessor:
    def __init__(self, encoder_type):
        self.encoder_type = encoder_type
        self.transform = None
        self.processor = None

        if encoder_type in ["resnet50", "efficientnet_b3"]:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif encoder_type == "vit_b_16":
            self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        elif encoder_type == "clip_vit_b_16":
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def __call__(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            return self.transform(image)
        return self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)


def preprocess_single_image(image_path, encoder_type):
    return ImagePreprocessor(encoder_type)(image_path)

def generate_caption(model, image_tensor, tokenizer, device, max_length=32):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    generated = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_length - 1):
            attention_mask = torch.ones_like(generated, device=device)
            _, logits = model(
                pixel_values=image_tensor,
                input_ids=generated,
                attention_mask=attention_mask
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated.squeeze(0).detach().cpu().tolist())


def generate_caption_beam_search(model, image_tensor, tokenizer, device,
                                 max_length=32, beam_size=3, length_penalty=0.7):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    beams = [([tokenizer.bos_token_id], 0.0)]
    completed = []

    with torch.inference_mode():
        for _ in range(max_length - 1):
            candidates = []
            for token_ids, score in beams:
                if token_ids[-1] == tokenizer.eos_token_id:
                    completed.append((token_ids, score))
                    continue

                input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids, device=device)
                _, logits = model(
                    pixel_values=image_tensor,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                top_scores, top_ids = torch.topk(log_probs, k=beam_size)

                for token_score, token_id in zip(top_scores.tolist(), top_ids.tolist()):
                    candidates.append((token_ids + [int(token_id)], score + float(token_score)))

            if not candidates:
                break

            def normalized_score(item):
                token_ids, score = item
                length = max(1, len(token_ids) - 1)
                return score / (length ** length_penalty)

            beams = sorted(candidates, key=normalized_score, reverse=True)[:beam_size]
            if all(token_ids[-1] == tokenizer.eos_token_id for token_ids, _ in beams):
                completed.extend(beams)
                break

    best_tokens, _ = max(completed or beams, key=lambda item: item[1] / (max(1, len(item[0]) - 1) ** length_penalty))
    return tokenizer.decode(best_tokens)


def generate_captions_batch(model, image_tensors, tokenizer, device, max_length=32):
    model.eval()
    pixel_values = torch.stack(image_tensors).to(device)
    batch_size = pixel_values.size(0)
    generated = torch.full(
        (batch_size, 1),
        tokenizer.bos_token_id,
        dtype=torch.long,
        device=device
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with torch.inference_mode():
        for _ in range(max_length - 1):
            attention_mask = torch.ones_like(generated, device=device)
            _, logits = model(
                pixel_values=pixel_values,
                input_ids=generated,
                attention_mask=attention_mask
            )
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, tokenizer.pad_token_id),
                next_token
            )
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(tokenizer.eos_token_id)
            if finished.all():
                break

    return [
        tokenizer.decode(token_ids)
        for token_ids in generated.detach().cpu().tolist()
    ]

def generate_captions_from_folder(model, test_img_dir, tokenizer, encoder_type, device,
                                  test_csv=None, max_length=32, output_csv="test_predictions.csv",
                                  batch_size=16, decode_strategy="greedy",
                                  beam_size=3, length_penalty=0.7):
    test_img_dir = Path(test_img_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    references_by_image = {}
    if test_csv is not None and os.path.exists(test_csv):
        ref_df = pd.read_csv(test_csv)
        if "image" not in ref_df.columns or "caption" not in ref_df.columns:
            ref_df.columns = ["image", "caption"]
        for image_name, group in ref_df.groupby("image"):
            references_by_image[str(image_name)] = group["caption"].astype(str).tolist()
        image_names = list(references_by_image.keys())
    else:
        image_names = sorted([p.name for p in test_img_dir.iterdir() if p.suffix.lower() in image_exts])

    rows = []
    batch_names = []
    batch_tensors = []
    preprocessor = ImagePreprocessor(encoder_type)

    def flush_batch():
        if not batch_tensors:
            return
        if decode_strategy == "beam":
            pred_captions = [
                generate_caption_beam_search(
                    model=model,
                    image_tensor=image_tensor,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=max_length,
                    beam_size=beam_size,
                    length_penalty=length_penalty
                )
                for image_tensor in batch_tensors
            ]
        elif decode_strategy == "greedy":
            pred_captions = generate_captions_batch(
                model=model,
                image_tensors=batch_tensors,
                tokenizer=tokenizer,
                device=device,
                max_length=max_length
            )
        else:
            raise ValueError(f"Unknown decode_strategy: {decode_strategy}")
        for batch_image_name, pred_caption in zip(batch_names, pred_captions):
            rows.append({
                "image": batch_image_name,
                "generated_caption": pred_caption,
                "reference_captions": " | ".join(references_by_image.get(batch_image_name, []))
            })
        batch_names.clear()
        batch_tensors.clear()

    for image_name in tqdm(image_names, desc="Generating captions"):
        image_path = test_img_dir / image_name
        if not image_path.exists():
            print(f"[!] Missing image: {image_path}")
            continue
        batch_names.append(image_name)
        batch_tensors.append(preprocessor(image_path))
        if len(batch_tensors) >= batch_size:
            flush_batch()

    flush_batch()

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(output_csv, index=False)
    print(f"[*] Saved generated captions to: {output_csv}")
    return pred_df, references_by_image

def _metric_tokens(text):
    return re.findall(r"\b\w+\b", str(text).lower())

def _rouge_l_f1(reference_tokens, hypothesis_tokens):
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    dp = [[0] * (len(hypothesis_tokens) + 1) for _ in range(len(reference_tokens) + 1)]
    for i, ref_token in enumerate(reference_tokens, 1):
        for j, hyp_token in enumerate(hypothesis_tokens, 1):
            if ref_token == hyp_token:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    precision = lcs / len(hypothesis_tokens)
    recall = lcs / len(reference_tokens)
    return 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)


class _NoWordNet:
    def synsets(self, *_args, **_kwargs):
        return []


def _safe_meteor_score(references, hypothesis):
    try:
        return meteor_score(references, hypothesis)
    except LookupError:
        return meteor_score(references, hypothesis, wordnet=_NoWordNet())


def evaluate_caption_metrics(pred_df, references_by_image):
    if corpus_bleu is None or meteor_score is None:
        raise ImportError("Please install nltk first: pip install nltk")

    eval_rows = []
    for row in pred_df.itertuples(index=False):
        refs = references_by_image.get(row.image, [])
        if refs:
            eval_rows.append((row.image, refs, row.generated_caption))

    if not eval_rows:
        print("[!] No reference captions found. Metrics skipped; generated captions are still saved.")
        return {}

    refs_tokenized = [[_metric_tokens(ref) for ref in refs] for _, refs, _ in eval_rows]
    hyps_tokenized = [_metric_tokens(pred) for _, _, pred in eval_rows]
    smooth = SmoothingFunction().method1

    bleu_1 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu_4 = corpus_bleu(refs_tokenized, hyps_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    meteor = sum(_safe_meteor_score(refs, hyp) for refs, hyp in zip(refs_tokenized, hyps_tokenized)) / len(eval_rows)
    rouge_l = sum(max(_rouge_l_f1(_metric_tokens(ref), hyp) for ref in refs)
                  for (_, refs, _), hyp in zip(eval_rows, hyps_tokenized)) / len(eval_rows)

    cider = None
    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {image: refs for image, refs, _ in eval_rows}
        res = {image: [pred] for image, _, pred in eval_rows}
        cider, _ = Cider().compute_score(gts, res)
    except Exception as exc:
        print(f"[!] CIDEr skipped. Install pycocoevalcap to enable it. Reason: {exc}")

    metrics = {
        "BLEU-1": bleu_1,
        "BLEU-4": bleu_4,
        "METEOR": meteor,
        "ROUGE-L": rouge_l,
        "CIDEr": cider
    }
    print(pd.DataFrame([metrics]).to_string(index=False))
    return metrics

def build_tokenizer_from_csv(train_csv, min_freq=2):
    if not train_csv or not os.path.exists(train_csv):
        raise FileNotFoundError(
            "Tokenizer is not saved in the notebook, so evaluate.py must rebuild it "
            f"from the training captions CSV. Missing file: {train_csv}"
        )

    train_df = pd.read_csv(train_csv)
    if "image" not in train_df.columns or "caption" not in train_df.columns:
        train_df.columns = ["image", "caption"]

    train_captions = train_df["caption"].astype(str).tolist()
    return CustomTokenizer(captions=train_captions, min_freq=min_freq)

def load_selected_captioner(checkpoint_path, encoder_type, tokenizer, device=None,
                            d_model=512, nhead=8, num_layers=6):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversalTransformerCaptioner(
        encoder_type=encoder_type,
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    model.to(device)
    load_model(model, checkpoint_path, device=device)
    model.eval()
    return model, device


# -----------------------------
# CONFIG VARIABLES
# -----------------------------
# Set USE_CONFIG_VARIABLES = True to run by editing these variables.
# Set it to False to use command-line arguments instead.
USE_CONFIG_VARIABLES = True

ENCODER_TYPE = "vit_b_16"
CHECKPOINT = "vit_b_16/best_captioner.pth"
TRAIN_CSV = "train_captions.csv"
TEST_IMG_DIR = "Images"
TEST_CSV = "test_captions.csv"
OUTPUT_CSV = 'test_predictions_vit_b_16.csv'

MAX_LENGTH = 32
BATCH_SIZE = 16
DECODE_STRATEGY = "beam"  # "greedy" or "beam"
BEAM_SIZE = 3
LENGTH_PENALTY = 0.7
MIN_FREQ = 2
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 6
DEVICE = None
SKIP_METRICS = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image captioning checkpoints.")
    parser.add_argument(
        "--encoder-type",
        default=ENCODER_TYPE,
        choices=["resnet50", "efficientnet_b3", "vit_b_16", "clip_vit_b_16"],
        help="Encoder used when training the checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to best_captioner.pth/last_captioner.pth. Defaults to checkpoints/<encoder-type>/best_captioner.pth."
    )
    parser.add_argument(
        "--train-csv",
        default=TRAIN_CSV,
        help="Training captions CSV used to rebuild the same CustomTokenizer vocabulary."
    )
    parser.add_argument(
        "--test-img-dir",
        default=TEST_IMG_DIR,
        help="Folder containing test images."
    )
    parser.add_argument(
        "--test-csv",
        default=TEST_CSV,
        help="Optional CSV with image,caption references for metrics."
    )
    parser.add_argument(
        "--output-csv",
        default=OUTPUT_CSV,
        help="Where to save generated captions."
    )
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--decode-strategy", choices=["greedy", "beam"], default=DECODE_STRATEGY)
    parser.add_argument("--beam-size", type=int, default=BEAM_SIZE)
    parser.add_argument("--length-penalty", type=float, default=LENGTH_PENALTY)
    parser.add_argument("--min-freq", type=int, default=MIN_FREQ)
    parser.add_argument("--d-model", type=int, default=D_MODEL)
    parser.add_argument("--nhead", type=int, default=NHEAD)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--device", default=DEVICE, help="cuda, cpu, or leave empty for auto.")
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Only generate captions; do not compute BLEU/METEOR/ROUGE/CIDEr."
    )
    return parser.parse_args()


def resolve_checkpoint_path(encoder_type, checkpoint=None):
    if checkpoint:
        return checkpoint

    candidates = [
        os.path.join("checkpoints", encoder_type, "best_captioner.pth"),
        os.path.join(encoder_type, "best_captioner.pth"),
        os.path.join("checkpoints", encoder_type, "last_captioner.pth"),
        os.path.join(encoder_type, "last_captioner.pth"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def run_evaluation(encoder_type, checkpoint, train_csv, test_img_dir, test_csv,
                   output_csv, max_length=32, batch_size=16, min_freq=2,
                   d_model=512, nhead=8, num_layers=6, device=None,
                   skip_metrics=False, decode_strategy="greedy",
                   beam_size=3, length_penalty=0.7):
    checkpoint = resolve_checkpoint_path(encoder_type, checkpoint)

    tokenizer = build_tokenizer_from_csv(train_csv, min_freq=min_freq)
    checkpoint_vocab_size = get_checkpoint_vocab_size(checkpoint)
    if checkpoint_vocab_size is not None and checkpoint_vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Tokenizer vocabulary size does not match checkpoint. "
            f"tokenizer.vocab_size={tokenizer.vocab_size}, "
            f"checkpoint vocab_size={checkpoint_vocab_size}. "
            "Use the exact train CSV/min_freq from training, or load the saved tokenizer."
        )

    model, device = load_selected_captioner(
        checkpoint_path=checkpoint,
        encoder_type=encoder_type,
        tokenizer=tokenizer,
        device=device,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    )
    pred_df, refs = generate_captions_from_folder(
        model=model,
        test_img_dir=test_img_dir,
        tokenizer=tokenizer,
        encoder_type=encoder_type,
        device=device,
        test_csv=test_csv,
        max_length=max_length,
        output_csv=output_csv,
        batch_size=batch_size,
        decode_strategy=decode_strategy,
        beam_size=beam_size,
        length_penalty=length_penalty
    )
    if not skip_metrics:
        evaluate_caption_metrics(pred_df, refs)
    return pred_df, refs


def main():
    # if USE_CONFIG_VARIABLES:
    #     run_evaluation(
    #         encoder_type=ENCODER_TYPE,
    #         checkpoint=CHECKPOINT,
    #         train_csv=TRAIN_CSV,
    #         test_img_dir=TEST_IMG_DIR,
    #         test_csv=TEST_CSV,
    #         output_csv=OUTPUT_CSV,
    #         max_length=MAX_LENGTH,
    #         batch_size=BATCH_SIZE,
    #         decode_strategy=DECODE_STRATEGY,
    #         beam_size=BEAM_SIZE,
    #         length_penalty=LENGTH_PENALTY,
    #         min_freq=MIN_FREQ,
    #         d_model=D_MODEL,
    #         nhead=NHEAD,
    #         num_layers=NUM_LAYERS,
    #         device=DEVICE,
    #         skip_metrics=SKIP_METRICS
    #     )
    #     return

    # args = parse_args()
    # run_evaluation(
    #     encoder_type=args.encoder_type,
    #     checkpoint=args.checkpoint,
    #     train_csv=args.train_csv,
    #     test_img_dir=args.test_img_dir,
    #     test_csv=args.test_csv,
    #     output_csv=args.output_csv,
    #     max_length=args.max_length,
    #     batch_size=args.batch_size,
    #     decode_strategy=args.decode_strategy,
    #     beam_size=args.beam_size,
    #     length_penalty=args.length_penalty,
    #     min_freq=args.min_freq,
    #     d_model=args.d_model,
    #     nhead=args.nhead,
    #     num_layers=args.num_layers,
    #     device=args.device,
    #     skip_metrics=args.skip_metrics
    # )
    ENCODER_TYPE = "resnet50"  # "resnet50", "efficientnet_b3", "vit_b_16", "clip_vit_b_16"
    CHECKPOINT = "resnet50/best_captioner.pth"
    TRAIN_CSV = "train_captions.csv"
    IMAGE_PATH = "Images/1299459550_1fd5594fa2.jpg"

    tokenizer = build_tokenizer_from_csv(TRAIN_CSV, min_freq=2)

    model, device = load_selected_captioner(
        checkpoint_path=CHECKPOINT,
        encoder_type=ENCODER_TYPE,
        tokenizer=tokenizer
    )

    image_tensor = preprocess_single_image(IMAGE_PATH, ENCODER_TYPE)

    caption = generate_caption_beam_search(
        model=model,
        image_tensor=image_tensor,
        tokenizer=tokenizer,
        device=device,
        max_length=32,
        beam_size=3
    )

    print(caption)

if __name__ == "__main__":
    main()
