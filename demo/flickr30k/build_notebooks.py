"""Dựng 2 notebook Flickr30k từ 2 notebook COCO 2014 trong research/notebooks:

  clipvit-transformer-coco2014.ipynb                          -> flickr30k-clipvit-transformer.ipynb
  coco2014-clip-vit-dynlogits-contrastive-attnpool.ipynb      -> flickr30k-clip-vit-dynlogits-contrastive-attnpool.ipynb

Kiến trúc, hàm mất mát và vòng huấn luyện giữ nguyên; chỉ đổi phần dữ liệu (Flickr30k, Karpathy split) và thêm cell
đánh giá trên tập test sau khi train xong. Chạy:  python build_notebooks.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "research" / "notebooks"
CHAIR_SYNONYMS = (HERE.parent / "chair_synonyms.txt").read_text(encoding="utf-8").strip()
assert '"""' not in CHAIR_SYNONYMS


def sub_block(src, start, end, new, include_end=False):
    """Thay đoạn từ `start` tới `end` (không gồm `end` trừ khi include_end) bằng `new`; cả hai mốc phải duy nhất."""
    assert src.count(start) == 1, ("start", start[:60], src.count(start))
    a = src.index(start)
    assert src.count(end, a) >= 1, ("end", end[:60])
    b = src.index(end, a) + (len(end) if include_end else 0)
    return src[:a] + new + src[b:]


def rep(src, old, new, count=1):
    assert src.count(old) == count, (old[:70], src.count(old))
    return src.replace(old, new)


# ------------------------------------------------------------------ khối dữ liệu Flickr30k (dùng chung)
PATHS_BLOCK = '''# Đường dẫn Flickr30k trên Kaggle. None = tự dò dưới /kaggle/input, nên chỉ cần gắn (Add Input) 2 dataset:
#   1) Karpathy split của Flickr30k (file dataset_flickr30k.json: train 29.000 / val 1.014 / test 1.000 ảnh, 5 câu mỗi ảnh),
#   2) ảnh Flickr30k (thư mục chứa các file .jpg, vd 1000092795.jpg).
# Có thể điền tay nếu dò sai.
KARPATHY_JSON_FILE = None
FLICKR_IMG_DIR = None
# Dự phòng khi không có file Karpathy: captions.txt (cột image,caption) hoặc results.csv (image_name| comment_number| comment).
# Khi đó ảnh được chia ngẫu nhiên (seed cố định) 29.000 / 1.014 / 1.000, KHÔNG trùng split Karpathy trong các bài báo.
FLICKR_CAPTIONS_FILE = None
# Flickr30k Karpathy không có restval: tập huấn luyện chỉ gồm "train".
TRAIN_SPLITS = ("train",)
'''

LOADERS_BLOCK = '''# ==========================================
# DỮ LIỆU FLICKR30K (Karpathy split)
# ==========================================
_FLICKR_META = None   # list[(filename, split, [câu gốc])]
SPLIT_SOURCE = None


def _find_first(root, predicate):
    for dirpath, _, files in os.walk(root):
        for name in files:
            if predicate(name):
                return os.path.join(dirpath, name)
    return None


def _load_flickr_meta():
    """Đọc danh sách (filename, split, câu gốc) một lần: ưu tiên dataset_flickr30k.json, nếu không có thì dùng file caption + chia ngẫu nhiên."""
    global _FLICKR_META, SPLIT_SOURCE, KARPATHY_JSON_FILE, FLICKR_IMG_DIR, FLICKR_CAPTIONS_FILE
    if _FLICKR_META is not None:
        return _FLICKR_META

    if KARPATHY_JSON_FILE is None and FLICKR_CAPTIONS_FILE is None:
        KARPATHY_JSON_FILE = _find_first("/kaggle/input", lambda n: n == "dataset_flickr30k.json")
        if KARPATHY_JSON_FILE is None:
            FLICKR_CAPTIONS_FILE = (_find_first("/kaggle/input", lambda n: n == "captions.txt")
                                    or _find_first("/kaggle/input", lambda n: n == "results.csv"))
    meta = []
    if KARPATHY_JSON_FILE:
        with open(KARPATHY_JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for img in data.get("images", []):
            sents = [s["raw"] if "raw" in s else " ".join(s.get("tokens", [])) for s in img.get("sentences", [])]
            if img.get("filename") and sents:
                meta.append((img["filename"], img.get("split"), sents))
        SPLIT_SOURCE = f"Karpathy split ({KARPATHY_JSON_FILE})"
    elif FLICKR_CAPTIONS_FILE:
        by_image = {}
        with open(FLICKR_CAPTIONS_FILE, "r", encoding="utf-8") as f:
            header = f.readline()
            sep = "|" if "|" in header else ","
            for line in f:
                parts = line.rstrip("\\n").split(sep, 2 if sep == "|" else 1)
                if len(parts) < 2:
                    continue
                name, cap = parts[0].strip(), parts[-1].strip()
                by_image.setdefault(name, []).append(cap)
        names = sorted(by_image)
        random.Random(SEED).shuffle(names)
        for i, name in enumerate(names):
            split = "train" if i < 29000 else "val" if i < 29000 + 1014 else "test" if i < 29000 + 1014 + 1000 else None
            if split:
                meta.append((name, split, by_image[name]))
        SPLIT_SOURCE = f"CHIA NGẪU NHIÊN từ {FLICKR_CAPTIONS_FILE} (không phải split Karpathy)"
    else:
        raise RuntimeError("Không tìm thấy dataset_flickr30k.json hay captions.txt/results.csv dưới /kaggle/input. "
                           "Hãy gắn dataset Flickr30k hoặc điền KARPATHY_JSON_FILE / FLICKR_CAPTIONS_FILE.")

    if FLICKR_IMG_DIR is None:
        probe = meta[0][0]
        hit = _find_first("/kaggle/input", lambda n: n == probe)
        if hit is None:
            raise RuntimeError(f"Không thấy ảnh {probe} dưới /kaggle/input. Hãy gắn dataset ảnh Flickr30k hoặc điền FLICKR_IMG_DIR.")
        FLICKR_IMG_DIR = os.path.dirname(hit)
    _FLICKR_META = meta
    return meta


def load_flickr_data(split="train"):
    """Trả về (image_paths, captions) theo cặp ảnh-caption; split là một tên hoặc tuple tên."""
    splits = {split} if isinstance(split, str) else set(split)
    image_paths, captions = [], []
    missing = 0
    for filename, sp, sents in _load_flickr_meta():
        if sp not in splits:
            continue
        path = os.path.join(FLICKR_IMG_DIR, filename)
        if not os.path.exists(path):
            missing += 1
            continue
        for raw in sents:
            caption = custom_standardization(raw)
            if caption:
                image_paths.append(path)
                captions.append(caption)
    print(f"-> Flickr30k {sorted(splits)}: {len(set(image_paths))} ảnh, {len(captions)} cặp ảnh-caption.")
    if missing:
        print(f"-> Bỏ qua {missing} ảnh vì không tìm thấy file ảnh.")
    return image_paths, captions


'''

CAPTION_OBJECTS_BLOCK = '''def build_caption_objects(image_paths, captions):
    """Flickr30k không có COCO instances nên không có nhãn vật thể do người vẽ. Thay bằng nhãn rút từ chính 5 caption của ảnh:
    object (trong 80 lớp COCO) được coi là CÓ trong ảnh nếu ít nhất 1 caption của ảnh nhắc tới (qua OBJECT_WORDS, gồm số nhiều).
    Hệ quả cần nhớ: vật thể thấy được nhưng không caption nào nhắc sẽ bị coi là vắng mặt, nên Dynamic Logits và negative sẽ phạt
    nhầm nhiều hơn so với nhãn COCO instances. Khóa là tên file, giống bản COCO."""
    word_to_category = {}
    for category, words in OBJECT_WORDS.items():
        for w in words:
            word_to_category.setdefault(w, category)
    image_objects = {}
    for path, cap in zip(image_paths, captions):
        found = image_objects.setdefault(os.path.basename(path), set())
        for w in cap.split():
            category = word_to_category.get(w)
            if category is not None:
                found.add(category)
    print(f"-> Đã rút nhãn object từ caption cho {len(image_objects)} ảnh "
          f"(trung bình {sum(len(v) for v in image_objects.values()) / max(len(image_objects), 1):.2f} object/ảnh).")
    return image_objects

'''

# ------------------------------------------------------------------ cell đánh giá (dùng chung)
EVAL_MARKDOWN = '''## Đánh giá sau khi train (tập test Karpathy, 1.000 ảnh)

Cell kế tiếp nạp **best model** (`BEST_MODEL_FILE`), sinh câu cho tập test bằng greedy và beam search rồi tính:

- **Chất lượng câu**: BLEU-1..4, METEOR (cần Java), ROUGE-L, CIDEr; SPICE tùy chọn (`EVAL_USE_SPICE`, chậm, cần Java và tải Stanford CoreNLP).
- **Ảo giác đối tượng (CHAIRs / CHAIRi)**: dùng bảng đồng nghĩa CHAIR gốc (Rohrbach et al., 2018). Nhãn đúng lấy từ 5 caption tham chiếu của ảnh
  vì Flickr30k không có COCO instances, nên con số cao hơn và **không so trực tiếp được** với CHAIR trên COCO.

Kết quả lưu ở `EVAL_RESULT_FILE` (JSON). Nếu kernel đã khởi động lại, chạy lại các cell định nghĩa (cấu hình, dữ liệu, mô hình) rồi chạy cell này; cell tự nạp từ điển và checkpoint từ `/kaggle/working`.
'''

EVAL_CODE = '''# ==========================================
# ĐÁNH GIÁ SAU KHI TRAIN: tập test Karpathy của Flickr30k
# ==========================================
import shutil
import time
import torch.nn.functional as F

EVAL_SPLIT = "test"
EVAL_CHECKPOINT = BEST_MODEL_FILE   # đổi thành CHECKPOINT_FILE để đánh giá epoch cuối
EVAL_BEAM_SIZE = 3
EVAL_LENGTH_ALPHA = 0.7             # chọn câu beam theo logprob / độ dài^alpha
EVAL_MAX_IMAGES = None              # None = toàn bộ; đặt số nhỏ (vd 50) để chạy thử nhanh
EVAL_BATCH_SIZE = 50
EVAL_USE_METEOR = True              # cần Java; thiếu Java thì tự bỏ qua
EVAL_USE_SPICE = False              # cần Java + tải Stanford CoreNLP, chậm (hàng chục phút với 1.000 ảnh)
EVAL_RESULT_FILE = os.path.join(OUTPUT_DIR, "__RESULT_NAME__")

CHAIR_SYNONYMS_TEXT = """__CHAIR_SYNONYMS__"""
_CHAIR_DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light", "traffic signal",
    "stop light", "fire hydrant", "stop sign", "parking meter", "suit case", "sports ball",
    "baseball bat", "baseball glove", "tennis racket", "wine glass", "hot dog", "cell phone",
    "mobile phone", "teddy bear", "hair drier", "potted plant", "bow tie", "laptop computer",
    "stove top oven", "home plate", "train track",
]


def _load_chair_tables():
    objects, inverse = set(), {}
    for line in CHAIR_SYNONYMS_TEXT.splitlines():
        syns = [s.strip() for s in line.split(",") if s.strip()]
        if not syns:
            continue
        objects.update(syns)
        for s in syns:
            inverse[s] = syns[0]
    double = {w: w for w in _CHAIR_DOUBLE_WORDS}
    for w in ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal", "cub"]:
        double[f"baby {w}"] = w
        double[f"adult {w}"] = w
    for w in ["jet", "train"]:
        double[f"passenger {w}"] = w
    double.update({"bow tie": "tie", "toilet seat": "toilet", "wine glas": "wine glass"})
    return objects, inverse, double


_CHAIR_OBJECTS, _CHAIR_INVERSE, _CHAIR_DOUBLE = _load_chair_tables()
try:
    from nltk.stem import WordNetLemmatizer
    _LEMMA = WordNetLemmatizer()
    _LEMMA.lemmatize("dogs", pos="n")
except Exception:
    try:
        import nltk
        nltk.download("wordnet", quiet=True)
        from nltk.stem import WordNetLemmatizer
        _LEMMA = WordNetLemmatizer()
        _LEMMA.lemmatize("dogs", pos="n")
    except Exception:
        _LEMMA = None  # không có wordnet: dùng luật số nhiều đơn giản bên dưới


def _singular(word):
    if _LEMMA is not None:
        return _LEMMA.lemmatize(word, pos="n")
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def chair_objects(caption):
    """Các lớp COCO (từ chuẩn) được caption nhắc tới, theo cách chair.py của CHAIR."""
    words = [_singular(w) for w in caption.lower().split()]
    merged, i = [], 0
    while i < len(words):
        two = " ".join(words[i:i + 2])
        if two in _CHAIR_DOUBLE:
            merged.append(_CHAIR_DOUBLE[two])
            i += 2
        else:
            merged.append(words[i])
            i += 1
    if "toilet" in merged and "seat" in merged:
        merged = [w for w in merged if w != "seat"]
    return [_CHAIR_INVERSE[w] for w in merged if w in _CHAIR_OBJECTS]


def compute_chair(preds, refs_list):
    """CHAIRs: % câu nhắc ít nhất 1 object không có trong 5 caption tham chiếu; CHAIRi: % lượt nhắc object bị ảo giác."""
    n_caps = n_hall_caps = n_mentions = n_hall = skipped = 0
    for pred, refs in zip(preds, refs_list):
        gt = set()
        for r in refs:
            gt.update(chair_objects(r))
        if not gt:
            skipped += 1
            continue
        mentioned = chair_objects(pred)
        bad = [o for o in mentioned if o not in gt]
        n_caps += 1
        n_mentions += len(mentioned)
        n_hall += len(bad)
        n_hall_caps += bool(bad)
    return {"CHAIRs": 100.0 * n_hall_caps / max(n_caps, 1), "CHAIRi": 100.0 * n_hall / max(n_mentions, 1),
            "num_captions": n_caps, "skipped_no_ground_truth": skipped}


def _ids_to_text(ids, idx2word):
    out = []
    for t in ids:
        if t == END_IDX:
            break
        if t in (START_IDX, PAD_IDX):
            continue
        out.append(idx2word.get(t, "<unk>"))
    return " ".join(out)


@torch.no_grad()
def decode_greedy(model_, images):
    memory = model_.encoder(images)
    B = images.size(0)
    seqs = torch.full((B, 1), START_IDX, dtype=torch.long, device=images.device)
    done = torch.zeros(B, dtype=torch.bool, device=images.device)
    for _ in range(SEQ_LENGTH - 1):
        mask = model_.generate_square_subsequent_mask(seqs.size(1)).to(images.device)
        out = model_.transformer_decoder(tgt=model_.pos_encoder(model_.embedding(seqs)), memory=memory, tgt_mask=mask)
        nxt = model_.fc_out(out[:, -1]).argmax(-1).masked_fill(done, PAD_IDX)
        seqs = torch.cat([seqs, nxt.unsqueeze(1)], dim=1)
        done = done | (nxt == END_IDX)
        if done.all():
            break
    return seqs.tolist()


@torch.no_grad()
def decode_beam(model_, images, beam=3, alpha=0.7):
    memory = model_.encoder(images)
    B, L, D = memory.shape
    mem = memory.unsqueeze(1).expand(-1, beam, -1, -1).reshape(B * beam, L, D)
    seqs = torch.full((B * beam, 1), START_IDX, dtype=torch.long, device=images.device)
    scores = torch.zeros(B, beam, device=images.device)
    scores[:, 1:] = -1e9   # bước đầu chỉ giữ 1 nhánh để các beam không trùng nhau
    done = torch.zeros(B * beam, dtype=torch.bool, device=images.device)
    base = (torch.arange(B, device=images.device) * beam).unsqueeze(1)
    for _ in range(SEQ_LENGTH - 1):
        mask = model_.generate_square_subsequent_mask(seqs.size(1)).to(images.device)
        out = model_.transformer_decoder(tgt=model_.pos_encoder(model_.embedding(seqs)), memory=mem, tgt_mask=mask)
        logp = F.log_softmax(model_.fc_out(out[:, -1]).float(), dim=-1)
        V = logp.size(-1)
        logp[done] = -1e9
        logp[done, PAD_IDX] = 0.0   # nhánh đã kết thúc chỉ được nối <pad>, không đổi điểm
        cand = (scores.reshape(-1, 1) + logp).reshape(B, beam * V)
        top_scores, top_idx = cand.topk(beam, dim=1)
        sel = (base + top_idx // V).reshape(-1)
        tok = (top_idx % V).reshape(-1)
        seqs = torch.cat([seqs[sel], tok.unsqueeze(1)], dim=1)
        done = done[sel] | (tok == END_IDX)
        scores = top_scores
        if done.all():
            break
    lengths = ((seqs != PAD_IDX) & (seqs != START_IDX)).sum(dim=1).clamp_min(1).float().reshape(B, beam)
    best = (scores / lengths.pow(alpha)).argmax(dim=1)
    return seqs.reshape(B, beam, -1)[torch.arange(B, device=images.device), best].tolist()


def compute_caption_metrics(preds, refs_list):
    """BLEU/METEOR/ROUGE-L/CIDEr(/SPICE) bằng pycocoevalcap trên chữ đã chuẩn hóa sẵn (không cần PTBTokenizer). Điểm nhân 100."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    gts = {i: refs for i, refs in enumerate(refs_list)}
    res = {i: [p if p else "<unk>"] for i, p in enumerate(preds)}
    metrics = {}
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    for k in range(4):
        metrics[f"BLEU-{k + 1}"] = 100 * bleu[k]
    has_java = shutil.which("java") is not None
    if EVAL_USE_METEOR and has_java:
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            metrics["METEOR"] = 100 * Meteor().compute_score(gts, res)[0]
        except Exception as exc:
            print(f"[!] Bỏ qua METEOR: {exc}")
    elif EVAL_USE_METEOR:
        print("[!] Không có Java trong PATH: bỏ qua METEOR.")
    metrics["ROUGE-L"] = 100 * Rouge().compute_score(gts, res)[0]
    metrics["CIDEr"] = 100 * Cider().compute_score(gts, res)[0]
    if EVAL_USE_SPICE and has_java:
        try:
            from pycocoevalcap.spice.spice import Spice
            metrics["SPICE"] = 100 * Spice().compute_score(gts, res)[0]
        except Exception as exc:
            print(f"[!] Bỏ qua SPICE: {exc}")
    return metrics


class _EvalDataset(Dataset):
    def __init__(self, paths, transforms):
        self.paths, self.transforms = paths, transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", IMAGE_SIZE, (0, 0, 0))
        return self.transforms(img)


def load_eval_model(checkpoint_file, vocab_size):
__BUILD_MODEL__
    path = _resolve_checkpoint_path(checkpoint_file)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    m.load_state_dict(ckpt.get("model_state_dict", ckpt))
    m.eval()
    print(f"[*] Đã nạp {path} (epoch {ckpt.get('epoch', -1) + 1 if isinstance(ckpt, dict) else '?'})")
    return m


def run_evaluation():
    eval_vocab = json.load(open(VOCAB_FILE, "r", encoding="utf-8"))
    idx2word = {i: w for w, i in eval_vocab.items()}
    paths, caps = load_flickr_data(EVAL_SPLIT)
    refs_by_path, order = {}, []
    for p, c in zip(paths, caps):
        if p not in refs_by_path:
            refs_by_path[p] = []
            order.append(p)
        refs_by_path[p].append(c)
    if EVAL_MAX_IMAGES:
        order = order[:EVAL_MAX_IMAGES]
    refs_list = [refs_by_path[p] for p in order]
    print(f"[*] Đánh giá {len(order)} ảnh của split '{EVAL_SPLIT}' bằng {os.path.basename(EVAL_CHECKPOINT)}")

    model_ = load_eval_model(EVAL_CHECKPOINT, len(eval_vocab))
    loader = DataLoader(_EvalDataset(order, val_transforms), batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=2)

    results, samples = {}, {}
    for name, fn in (("greedy", lambda x: decode_greedy(model_, x)),
                     (f"beam{EVAL_BEAM_SIZE}", lambda x: decode_beam(model_, x, EVAL_BEAM_SIZE, EVAL_LENGTH_ALPHA))):
        t0 = time.time()
        preds = []
        for images in loader:
            for ids in fn(images.to(device)):
                preds.append(_ids_to_text(ids, idx2word))
        decode_s = time.time() - t0
        metrics = compute_caption_metrics(preds, refs_list)
        metrics.update(compute_chair(preds, refs_list))
        metrics["mean_len"] = sum(len(p.split()) for p in preds) / max(len(preds), 1)
        metrics["unk_rate_%"] = 100 * sum(p.split().count("<unk>") for p in preds) / max(sum(len(p.split()) for p in preds), 1)
        metrics["decode_seconds"] = decode_s
        results[name] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in metrics.items()}
        samples[name] = [{"image": os.path.basename(p), "pred": c, "refs": r[:2]} for p, c, r in list(zip(order, preds, refs_list))[:10]]
        print(f"\\n=== {name} ===")
        for k, v in results[name].items():
            print(f"   {k:>16}: {v}")

    out = {"checkpoint": EVAL_CHECKPOINT, "split": EVAL_SPLIT, "split_source": SPLIT_SOURCE, "num_images": len(order),
           "beam_size": EVAL_BEAM_SIZE, "results": results, "samples": samples}
    with open(EVAL_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\\n[*] Đã lưu kết quả: {EVAL_RESULT_FILE}")
    return out


if __name__ == "__main__":
    eval_out = run_evaluation()
'''


def eval_cell(build_model, result_name):
    return (EVAL_CODE.replace("__BUILD_MODEL__", build_model).replace("__RESULT_NAME__", result_name)
            .replace("__CHAIR_SYNONYMS__", CHAIR_SYNONYMS))


def read_cells(name):
    nb = json.loads((SRC / name).read_text(encoding="utf-8"))
    return nb, ["".join(c["source"]) for c in nb["cells"]]


def make_cell(kind, text):
    lines = text.splitlines(keepends=True)
    cell = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def write_nb(path, cells, template_nb):
    nb = {"cells": [make_cell(k, t) for k, t in cells], "metadata": template_nb["metadata"],
          "nbformat": 4, "nbformat_minor": 4}
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, t in cells:
        if k == "code":
            compile(t, str(path.name), "exec")
    print("saved", path)


# ================================================================== notebook 1: baseline
nb1, c1 = read_cells("clipvit-transformer-coco2014.ipynb")
s = c1[0]
s = sub_block(s, "# Đường dẫn COCO 2014 cho baseline CLIP-ViT + Transformer.", 'TRAIN_SPLITS = ("train", "restval")\n', PATHS_BLOCK, include_end=True)
s = rep(s, "coco2014_karpathy_restval_baseline", "flickr30k_karpathy_baseline", 3)
s = sub_block(s, "# Train từ đầu với train+restval:", "RESUME_CHECKPOINT_FILE = CHECKPOINT_FILE\n",
              "# Train từ đầu. Nếu kernel bị ngắt giữa chừng, đổi RESUME_TRAINING=True để tiếp tục từ last_checkpoint của chính bản này.\n"
              "RESUME_TRAINING = False\nRESUME_CHECKPOINT_FILE = CHECKPOINT_FILE\n", include_end=True)
s = rep(s, "EPOCHS = 35\n", "EPOCHS = 30  # 1.512 bước/epoch (29.000 ảnh x 5 câu / 96); EarlyStopping (patience 3, val loss) thường dừng sớm hơn\n")
s = rep(s, "COCODataset", "FlickrDataset", 3)
s = sub_block(s, "def load_karpathy_data(json_file, split=\"train\"):", "def build_or_load_vocab(", LOADERS_BLOCK)
s = rep(s, 'print("0. Đang chuẩn bị Dataset và Vocabulary cho baseline COCO 2014...")', 'print("0. Đang chuẩn bị Dataset và Vocabulary cho baseline Flickr30k...")')
for line in ('    print(f"   TRAIN_IMG_DIR  : {TRAIN_IMG_DIR}")\n', '    print(f"   VAL_IMG_DIR    : {VAL_IMG_DIR}")\n', '    print(f"   KARPATHY_JSON  : {KARPATHY_JSON_FILE}")\n'):
    s = rep(s, line, "")
s = rep(s, "train_imgs, train_caps = load_karpathy_data(KARPATHY_JSON_FILE, split=TRAIN_SPLITS)\n",
        "train_imgs, train_caps = load_flickr_data(TRAIN_SPLITS)\n    print(f\"   Nguồn split: {SPLIT_SOURCE} | thư mục ảnh: {FLICKR_IMG_DIR}\")\n")
s = rep(s, 'val_imgs, val_caps = load_karpathy_data(KARPATHY_JSON_FILE, split="val")', 'val_imgs, val_caps = load_flickr_data("val")')
s = rep(s, "baseline CLIP-ViT-B/16 + Transformer trên COCO 2014...", "baseline CLIP-ViT-B/16 + Transformer trên Flickr30k...")
assert "COCO_" not in s and "TRAIN_IMG_DIR" not in s and "load_karpathy_data" not in s

md1 = """# CLIP-ViT-B/16 + Transformer baseline trên Flickr30k

Bản chuyển từ `clipvit-transformer-coco2014.ipynb` sang **Flickr30k (Karpathy split)**, kèm cell **đánh giá trên tập test sau khi train xong**.

- Kiến trúc, hàm mất mát và vòng huấn luyện giữ nguyên: CLIP ViT-B/16 đóng băng + adapter, bộ giải mã Transformer 2 lớp, AdamW, AMP, EarlyStopping theo val loss.
- Dữ liệu: `dataset_flickr30k.json` (train 29.000 / val 1.014 / test 1.000 ảnh, 5 câu mỗi ảnh), không có restval. Cần gắn 2 dataset (Karpathy split Flickr30k và ảnh Flickr30k); đường dẫn tự dò dưới `/kaggle/input`.
- `EPOCHS` = 30 (mỗi epoch 1.512 bước, ít hơn COCO khoảng 3,9 lần).
- Tên checkpoint/từ điển có tiền tố `flickr30k_karpathy_baseline_`.
"""

build1 = """    m = ImageCaptioningTransformer(vocab_size=vocab_size, embed_dim=EMBED_DIM, ff_dim=FF_DIM,
                                   num_heads=NUM_HEADS, num_layers=NUM_LAYERS).to(device)"""
cells1 = [("markdown", md1), ("code", s), ("markdown", EVAL_MARKDOWN),
          ("code", eval_cell(build1, "flickr30k_karpathy_baseline_eval_test.json"))]
write_nb(HERE / "flickr30k-clipvit-transformer.ipynb", cells1, nb1)

# ================================================================== notebook 2: DynLogits + contrastive + attention pool
nb2, c2 = read_cells("coco2014-clip-vit-dynlogits-contrastive-attnpool.ipynb")
cfg, model_cell, train_cell, main_cell = c2[1], c2[2], c2[3], c2[4]

cfg = sub_block(cfg, "# Đường dẫn COCO 2014\n", 'VAL_INSTANCES_JSON = "/kaggle/input/datasets/nadaibrahim/coco2014/captions/annotations/instances_val2014.json"\n',
                PATHS_BLOCK, include_end=True)
cfg = rep(cfg, "coco2014_karpathy_restval_dynlogits_contrastive_attnpool", "flickr30k_karpathy_dynlogits_contrastive_attnpool", 3)
cfg = sub_block(cfg, "# Hallucination-aware training: dùng COCO instances", "USE_HALL_LOSS = False",
                "# Hallucination-aware training: Flickr30k không có COCO instances, nên nhãn object của ảnh rút từ chính 5 caption\n"
                "# (xem build_caption_objects). Không dùng YOLO.\n")
cfg = rep(cfg, "USE_COCO_INSTANCE_HALL_LABELS = True\n", "USE_CAPTION_OBJECT_LABELS = True\n")
cfg = rep(cfg, "EPOCHS = 5  # theo yêu cầu: chỉ train 5 epoch cho ablation này (không phải 44 như bản N006/N009)",
          "EPOCHS = 30  # 1.512 bước/epoch (29.000 ảnh x 5 câu / 96); bản COCO chạy 40 epoch x 5.904 bước")
cfg = sub_block(cfg, "_DIR_FILES = {}", "def build_or_load_vocab(", LOADERS_BLOCK)
cfg = rep(cfg, "COCODataset", "FlickrDataset", 1)
assert "TRAIN_IMG_DIR" not in cfg and "INSTANCES_JSON" not in cfg and "COCO_" not in cfg.replace("COCO_OBJECT", "")

model_cell = sub_block(model_cell, "def load_coco_instance_objects(instances_file):", 'HALL_DEBUG_STATE = {"calls": 0}', CAPTION_OBJECTS_BLOCK)

m = main_cell
m = rep(m, 'print("0. Đang chuẩn bị Dataset và Vocabulary cho train-from-scratch COCO 2014...")', 'print("0. Đang chuẩn bị Dataset và Vocabulary cho train-from-scratch Flickr30k...")')
for line in ('    print(f"   TRAIN_IMG_DIR  : {TRAIN_IMG_DIR}")\n', '    print(f"   VAL_IMG_DIR    : {VAL_IMG_DIR}")\n', '    print(f"   KARPATHY_JSON  : {KARPATHY_JSON_FILE}")\n'):
    m = rep(m, line, "")
m = rep(m, "train_imgs, train_caps = load_karpathy_data(KARPATHY_JSON_FILE, split=TRAIN_SPLITS)\n",
        "train_imgs, train_caps = load_flickr_data(TRAIN_SPLITS)\n    print(f\"   Nguồn split: {SPLIT_SOURCE} | thư mục ảnh: {FLICKR_IMG_DIR}\")\n")
m = rep(m, 'val_imgs, val_caps = load_karpathy_data(KARPATHY_JSON_FILE, split="val")', 'val_imgs, val_caps = load_flickr_data("val")')
m = sub_block(m, "    val_image_objects = load_coco_instance_objects(VAL_INSTANCES_JSON)", "        train_image_objects = {**train_image_objects, **val_image_objects}\n",
              "    val_image_objects = build_caption_objects(val_imgs, val_caps) if (USE_OBJECT_GROUNDING and USE_CAPTION_OBJECT_LABELS) else {}\n"
              "    train_image_objects = build_caption_objects(train_imgs, train_caps) if (USE_OBJECT_GROUNDING and USE_CAPTION_OBJECT_LABELS) else {}\n",
              include_end=True)
m = rep(m, "USE_COCO_INSTANCE_HALL_LABELS", "USE_CAPTION_OBJECT_LABELS", 2)
m = rep(m, "COCO instances (ground truth only, không dùng YOLO)", "nhãn object rút từ 5 caption/ảnh (không dùng YOLO)")
m = sub_block(m, "            _restval_imgs = [p for p in train_imgs", "            diagnose_hall_label_coverage(val_imgs, val_image_objects", "")
m = rep(m, "COCODataset", "FlickrDataset", 2)
m = rep(m, "TỪ ĐẦU với hallucination-aware loss trên COCO 2014...", "TỪ ĐẦU với hallucination-aware loss trên Flickr30k...")
m = rep(m, "Thường do TRAIN_INSTANCES_JSON/VAL_INSTANCES_JSON sai path,", "Với Flickr30k, nhãn rút từ caption nên khớp gần như 100%; thấp nghĩa là parse caption lỗi,")
m = rep(m, "-> Kiểm tra lại TRAIN_INSTANCES_JSON/VAL_INSTANCES_JSON đúng dataset khớp với ảnh đang dùng (tiền tố COCO_train2014_/COCO_val2014_, đuôi .jpg, v.v).", "-> Kiểm tra lại cách đọc caption (load_flickr_data) và build_caption_objects.")
assert "TRAIN_IMG_DIR" not in m and "INSTANCES_JSON" not in m and "load_karpathy" not in m and "_restval" not in m

md2 = """# CLIP-ViT + Transformer — Dynamic Logits + Contrastive + Attention-Pooling trên Flickr30k

Bản chuyển từ `coco2014-clip-vit-dynlogits-contrastive-attnpool.ipynb` sang **Flickr30k (Karpathy split)**, kèm cell **đánh giá trên tập test sau khi train xong**.

Giữ nguyên kiến trúc và huấn luyện: Dynamic Logits tại đầu ra decoder (chặn mềm theo xác suất, warmup 10 epoch), contrastive đa dương tính (SupCon, 3 negative: 2 đổi object + 1 hoán vị trí kiểu NegCLIP), attention-pooling, label smoothing 0,1, cắt gradient 1,0, AMP.

## Khác so với bản COCO

1. **Dữ liệu**: `dataset_flickr30k.json` (train 29.000 / val 1.014 / test 1.000 ảnh, 5 câu mỗi ảnh), không có restval. Đường dẫn tự dò dưới `/kaggle/input`; cần gắn Karpathy split Flickr30k và ảnh Flickr30k.
2. **Nhãn object**: Flickr30k không có COCO instances, nên object "có trong ảnh" được rút từ 5 caption của ảnh (`build_caption_objects`, 80 lớp COCO). Vật thể thấy được nhưng không caption nào nhắc sẽ bị coi là vắng mặt, nên Dynamic Logits và negative phạt nhầm nhiều hơn bản COCO. Đây là chỗ cần xem kỹ khi đọc kết quả.
3. **Số epoch**: `EPOCHS` = 30 (mỗi epoch 1.512 bước, ít hơn COCO khoảng 3,9 lần).
4. **Đánh giá**: cell cuối chạy tự động sau khi train (greedy và beam search, BLEU/METEOR/ROUGE-L/CIDEr, CHAIRs/CHAIRi). Dynamic Logits ở đây chỉ dùng lúc train; lúc đánh giá không chặn logits (không có bằng chứng YOLO hay câu truy hồi).

Tên checkpoint/từ điển có tiền tố `flickr30k_karpathy_dynlogits_contrastive_attnpool_`.
"""

build2 = """    m = ImageCaptioningTransformer(vocab_size=vocab_size, embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS,
                                   num_layers=NUM_LAYERS, contrastive_proj_dim=CONTRASTIVE_PROJ_DIM).to(device)"""
cells2 = [("markdown", md2), ("code", cfg), ("code", model_cell), ("code", train_cell), ("code", m),
          ("markdown", EVAL_MARKDOWN),
          ("code", eval_cell(build2, "flickr30k_karpathy_dynlogits_contrastive_attnpool_eval_test.json"))]
write_nb(HERE / "flickr30k-clip-vit-dynlogits-contrastive-attnpool.ipynb", cells2, nb2)
