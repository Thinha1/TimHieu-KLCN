import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torchvision import transforms, models
from transformers import AutoTokenizer, CLIPProcessor, CLIPVisionModel, ViTModel, ViTImageProcessor
from tqdm import tqdm
import argparse

# Tối ưu hóa môi trường
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# =========================================================================
# 1. ENCODERS
# =========================================================================
class CNNEncoder(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        if model_name == "resnet50":
            # ResNet50
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            modules = list(resnet.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.out_dim = 2048
        elif model_name == "efficientnet_b3":
            # EfficientNet-B3
            effnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.model = effnet.features
            self.out_dim = 1536
        else:
            raise ValueError(f"Model {model_name} not supported by CNNEncoder")
            
        for param in self.model.parameters():
            param.requires_grad = False # Freeze encoder

    def forward(self, images):
        features = self.model(images) # [B, C, H, W]
        features = features.permute(0, 2, 3, 1) # [B, H, W, C]
        features = features.view(features.size(0), -1, features.size(-1)) # [B, H*W, C]
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

# =========================================================================
# 2. DATASET
# =========================================================================
class Flickr8kDataset(Dataset):
    def __init__(self, csv_path, img_dir,
                 encoder_type="resnet50",
                 tokenizer_name="gpt2",
                 max_length=50, is_train=True):
        self.img_dir = img_dir
        self.max_length = max_length
        self.is_train = is_train
        self.encoder_type = encoder_type

        # Text Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Image Processor
        if encoder_type in ["resnet50", "efficientnet_b3"]:
            # Standard Torchvision Transform
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if self.is_train:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize
                ])
            self.processor = None
        elif encoder_type == "vit_b_16":
            self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            self.transform = None
        elif encoder_type == "clip_vit_b_16":
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            self.transform = None
            
        # Optional augmentations for processor-based methods
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2) if self.is_train else None

        # Load Flickr8k captions.txt format: image,caption
        df = pd.read_csv(csv_path, sep=',')
        if 'image' not in df.columns or 'caption' not in df.columns:
            # Maybe it is image_name, comment
            df.columns = ['image', 'caption']
            
        self.image_files = df['image'].tolist()
        self.captions = df['caption'].tolist()

        print(f"[*] Dataset Flickr8k loaded: {len(self.image_files)} samples.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.encoder_type in ["resnet50", "efficientnet_b3"]:
            pixel_values = self.transform(image)
        else:
            if self.color_jitter:
                image = self.color_jitter(image)
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # Text padding
        caption_text = str(self.captions[idx])
        caption = self.tokenizer.bos_token + " " + caption_text + " " + self.tokenizer.eos_token
        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def collate_fn_custom(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }

# =========================================================================
# 3. MODEL ARCHITECTURE
# =========================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

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

        # 1. Encoder
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

        # Project output to transformer d_model
        self.enc_projection = nn.Linear(encoder_hidden_size, d_model)

        # 2. Transformer Decoder
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 3. Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, pixel_values, input_ids, labels=None, attention_mask=None):
        # 1. Encode Image
        with torch.no_grad():
            image_embeds = self.encoder(pixel_values)

        image_embeds = self.enc_projection(image_embeds)

        # 2. Decode Text
        seq_len = input_ids.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(input_ids.device)

        # Embedding + Positional Encoding
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

# =========================================================================
# 4. TRAINING LOOP
# =========================================================================
def _unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def save_model(model, optimizer=None, scaler=None, epoch=0, best_val_loss=None,
               save_path="captioner_checkpoint.pth"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": _unwrap_model(model).state_dict(),
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, save_path)
    print(f"[*] Đã lưu checkpoint: {save_path}")

def load_model(model, checkpoint_path, optimizer=None, scaler=None, device='cuda', strict=True):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0, float('inf')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        _unwrap_model(model).load_state_dict(checkpoint, strict=strict)
        return 0, float('inf')
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("best_val_loss", float('inf'))
    print(f"[*] Đã load checkpoint: {checkpoint_path} | resume từ epoch {start_epoch + 1}")
    return start_epoch, best_val_loss

def train_model(model, train_loader, val_loader=None, epochs=10, learning_rate=1e-4,
                device='cuda', checkpoint_dir="checkpoints", resume_from=None):
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    start_epoch = 0

    if resume_from:
        start_epoch, best_val_loss = load_model(
            model=model, checkpoint_path=resume_from,
            optimizer=optimizer, scaler=scaler, device=device
        )

    for epoch in range(start_epoch, epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16):
                loss, _ = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask
                )
            if loss.dim() > 0: loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            pbar.set_postfix({"Train Loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Val")
                for batch in val_pbar:
                    pixel_values = batch["pixel_values"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    with torch.autocast(device_type=device, dtype=torch.float16):
                        loss, _ = model(
                            pixel_values=pixel_values, input_ids=input_ids,
                            labels=labels, attention_mask=attention_mask
                        )
                    if loss.dim() > 0: loss = loss.mean()
                    total_val_loss += loss.item()
                    val_pbar.set_postfix({"Val Loss": f"{loss.item():.4f}"})

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_model(model, optimizer, scaler, epoch + 1, best_val_loss,
                           os.path.join(checkpoint_dir, "best_captioner.pth"))
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f}")

        save_model(model, optimizer, scaler, epoch + 1, best_val_loss,
                   os.path.join(checkpoint_dir, "last_captioner.pth"))

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # PARAMETERS / CONFIG
    # -------------------------------------------------------------------------
    # Lựa chọn 1 trong 4 models: 
    # 'resnet50', 'efficientnet_b3', 'vit_b_16', 'clip_vit_b_16'
    ENCODER_TYPE = "clip_vit_b_16"
    
    TRAIN_CSV = "path/to/flickr8k/captions.txt" 
    TRAIN_IMG_DIR = "path/to/flickr8k/Images"
    VAL_CSV = None # Tuỳ chọn
    VAL_IMG_DIR = None
    
    CHECKPOINT_DIR = "checkpoints"
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4

    # -------------------------------------------------------------------------
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TRAIN_IMG_DIR):
        print(f"⚠️ Vui lòng cập nhật đường dẫn TRAIN_CSV và TRAIN_IMG_DIR cho Flickr8k!")
    else:
        print(f"🚀 Khởi tạo model với encoder: {ENCODER_TYPE}")
        tokenizer_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        vocab_size = tokenizer.vocab_size

        train_dataset = Flickr8kDataset(
            csv_path=TRAIN_CSV,
            img_dir=TRAIN_IMG_DIR,
            encoder_type=ENCODER_TYPE,
            tokenizer_name=tokenizer_name,
            is_train=True
        )

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn_custom, pin_memory=True, num_workers=4
        )

        val_loader = None
        if VAL_CSV and VAL_IMG_DIR and os.path.exists(VAL_CSV):
            val_dataset = Flickr8kDataset(
                csv_path=VAL_CSV, img_dir=VAL_IMG_DIR,
                encoder_type=ENCODER_TYPE, tokenizer_name=tokenizer_name, is_train=False
            )
            val_loader = DataLoader(
                val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=collate_fn_custom, pin_memory=True, num_workers=4
            )

        model = UniversalTransformerCaptioner(
            encoder_type=ENCODER_TYPE,
            vocab_size=vocab_size,
            d_model=512,
            nhead=8,
            num_layers=6
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        if torch.cuda.device_count() > 1:
            print(f"🔥 Kích hoạt Multi-GPU (Tìm thấy {torch.cuda.device_count()} GPUs)")
            model = nn.DataParallel(model)

        print(f"Bắt đầu huấn luyện trên {device}...")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            device=device,
            checkpoint_dir=os.path.join(CHECKPOINT_DIR, ENCODER_TYPE)
        )
