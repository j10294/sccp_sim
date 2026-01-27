import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models


# ----------------------------
# Dataset
# ----------------------------
class NPZImageDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, train: bool, seed: int = 1):
        assert X.ndim == 4 and X.shape[1] == 3, f"Expected (N,3,H,W), got {X.shape}"
        self.X = X
        self.y = y.astype(np.int64)
        self.train = train
        self.rng = np.random.default_rng(seed)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return self.X.shape[0]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # x: (3, H, W), H=W=224

        _, H, W = x.shape

        # 1) Random Resized Crop (scale 0.6 ~ 1.0)
        scale = float(self.rng.uniform(0.6, 1.0))
        new_h = int(H * scale)
        new_w = int(W * scale)

        top = int(self.rng.integers(0, H - new_h + 1))
        left = int(self.rng.integers(0, W - new_w + 1))

        x = x[:, top:top + new_h, left:left + new_w]

        # resize back to 224x224
        x = F.interpolate(
            x.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # 2) Horizontal flip
        if self.rng.random() < 0.5:
            x = torch.flip(x, dims=[2])

        return x


    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float() / 255.0
        if self.train:
            x = self._augment(x)
        x = (x - self.mean) / self.std
        y = int(self.y[idx])
        return x, y


# ----------------------------
# Model
# ----------------------------
def build_model(name: str, num_classes: int, finetune: str='full'):
    """
    finetune: 'full', 'head', or 'last'
    - 'full' : full fine-tune
    - 'head' : train head only (freeze backbone)
    - 'last' : train head + last stage/block (partial fine-tune)
    """

    name = name.lower()
    finetune = finetune.lower()
    assert finetune in ['full','head','last']

    def set_requires_grad(module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    if name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

        if finetune == 'head':
            set_requires_grad(m, False)
            set_requires_grad(m.fc, True)
        elif finetune == 'last':
            set_requires_grad(m.conv1, False)
            set_requires_grad(m.bn1, False)
            set_requires_grad(m.layer4, True)
            set_requires_grad(m.fc, True)
        return m
    
    if name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

        if finetune == 'head':
            set_requires_grad(m, False)
            set_requires_grad(m.fc, True)
        elif finetune == 'last':
            set_requires_grad(m.conv1, False)
            set_requires_grad(m.bn1, False)
            set_requires_grad(m.layer4, True)
            set_requires_grad(m.fc, True)
        return m
    
    if name == 'mobilenet_v3_small':
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)


        if finetune == 'head':
            set_requires_grad(m, False)
            set_requires_grad(m.classifier, True)
        elif finetune == 'last':
            set_requires_grad(m, False)
            # heuristic : unfreeze last 2 inverted residual blocks
            if hasattr(m, 'features') and len(m.features) >= 2:
                set_requires_grad(m.features[-2], True)
            set_requires_grad(m.classifier, True)
        return m
    
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)

        if finetune == 'head':
            set_requires_grad(m, False)
            set_requires_grad(m.classifier, True)
        elif finetune == 'last':
            set_requires_grad(m, False)
            # heuristic : unfreeze last stage
            if hasattr(m, 'features') and len(m.features) >= 2:
                set_requires_grad(m.features[-2], True)
            set_requires_grad(m.classifier, True)
        return m
    

    if name == 'convnext_tiny':
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)

        if finetune == 'head':
            set_requires_grad(m, False)
            set_requires_grad(m.classifier, True)
        elif finetune == 'last':
            set_requires_grad(m, False)
            # convnext has 'features' (sequential of stages)
            if hasattr(m, 'features') and len(m.features) >= 1:
                set_requires_grad(m.features[-1], True) #last stage
            set_requires_grad(m.classifier, True)
        return m
    
    raise ValueError(f"Unknown model name: {name}")

# ----------------------------
# Train / Eval / Predict
# ----------------------------
@torch.no_grad()
def predict_proba(model, loader, device, amp: bool = True):
    model.eval()
    Ps, Ys = [], []

    use_amp = amp and device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            p = torch.softmax(logits, dim=1)

        Ps.append(p.detach().cpu().numpy())
        Ys.append(y.numpy())

    P = np.concatenate(Ps, axis=0)
    Y = np.concatenate(Ys, axis=0).astype(np.int64)
    return P, Y



def train_one_epoch(model, loader, opt, device, scaler, amp: bool = True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        loss_sum += float(loss.detach().cpu()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_acc(model, loader, device, amp: bool = True):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss_sum += float(loss.detach().cpu()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))
    return loss_sum / max(total, 1), correct / max(total, 1)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_npz", type=str, required=True)
    ap.add_argument("--out_prob_npz", type=str, required=True)

    ap.add_argument("--model", type=str, default="resnet50",
                    choices=["mobilenet_v3_small", "resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"])
    ap.add_argument("--finetune", type=str, default="head", choices=["full", "head", "last"],
                    help="full: all params, head: classifier only, last: last stage + head")
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="Freeze backbone (train classifier only). Default: full fine-tune.")

    ap.add_argument("--calibA_frac", type=float, default=0.5)
    ap.add_argument("--calib_split_seed", type=int, default=1)

    ap.add_argument("--amp", action="store_true", help="Use autocast + GradScaler.")
    ap.add_argument("--cudnn_benchmark", action="store_true", help="Enable cudnn benchmark (fixed image size).")
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true")

    ap.add_argument("--save_ckpt", type=str, default="")
    args = ap.parse_args()

    img_npz = os.path.expanduser(args.img_npz)
    out_prob_npz = os.path.expanduser(args.out_prob_npz)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and (device.type == "cuda")
    if device.type == "cuda" and args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    z = np.load(img_npz, allow_pickle=True)
    X_train, y_train = z["X_train"], z["y_train"]
    X_val, y_val     = z["X_val"], z["y_val"]
    X_test, y_test   = z["X_test"], z["y_test"]

    K = int(max(y_train.max(), y_val.max(), y_test.max())) + 1

    # Split val into Calib-A, Calib-B, Test2
    rng = np.random.default_rng(args.calib_split_seed)
    n_val = X_val.shape[0]
    perm = rng.permutation(n_val)

    nA = int(round(args.calibA_frac * n_val))
    remain = n_val - nA
    nB = remain // 2

    idxA = perm[:nA]
    idxB = perm[nA:nA + nB]
    idxT = perm[nA + nB:]

    X_calA, y_calA = X_val[idxA], y_val[idxA]
    X_calB, y_calB = X_val[idxB], y_val[idxB]
    X_test2, y_test2 = X_val[idxT], y_val[idxT]

    ds_train = NPZImageDataset(X_train, y_train, train=True, seed=args.seed)
    ds_valB  = NPZImageDataset(X_calB,  y_calB,  train=False, seed=args.seed)

    dl_common = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
    )
    if args.num_workers > 0:
        dl_common["prefetch_factor"] = args.prefetch_factor

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, **dl_common)
    dl_valB  = DataLoader(ds_valB,  batch_size=args.batch_size, shuffle=False, **dl_common)

    model = build_model(args.model, num_classes=K, finetune=args.finetune).to(device)

    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ('fc' in n) or ('classifier' in n):
            head_params.append(p)
        else:
            backbone_params.append(p)
    
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.lr_backbone})
    if head_params:
        param_groups.append({'params': head_params, 'lr': args.lr_head})

    opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    print(f"[device] {device}  [amp] {amp}")
    print(f"[model] {args.model}  [K] {K}  [freeze_backbone] {args.freeze_backbone}")
    print(f"[train] {X_train.shape}  [val] {X_val.shape} -> Calib-A {X_calA.shape}, Calib-B {X_calB.shape}, Test2 {X_test2.shape}")
    print(f"[test]  {X_test.shape}")

    n_trainable = sum(p.requires_grad for p in model.parameters())
    n_total = sum(1 for _ in model.parameters())
    print(f"[params] trainable tensors = {n_trainable} / total = {n_total}")

    best_val_acc = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, opt, device, scaler, amp=amp)
        va_loss, va_acc = eval_acc(model, dl_valB, device, amp=amp)
        print(f"[epoch {ep:02d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | Calib-B loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if args.save_ckpt:
        ckpt_path = os.path.expanduser(args.save_ckpt)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"model": args.model, "K": K, "state_dict": model.state_dict()}, ckpt_path)
        print(f"[saved ckpt] {ckpt_path}")

    # Probability extraction
    dl_calA = DataLoader(NPZImageDataset(X_calA, y_calA, train=False, seed=args.seed),
                         batch_size=args.batch_size, shuffle=False, **dl_common)
    dl_calB = DataLoader(NPZImageDataset(X_calB, y_calB, train=False, seed=args.seed),
                         batch_size=args.batch_size, shuffle=False, **dl_common)
    dl_test = DataLoader(NPZImageDataset(X_test2, y_test2, train=False, seed=args.seed),
                         batch_size=args.batch_size, shuffle=False, **dl_common)

    p_sel, y_sel = predict_proba(model, dl_calA, device, amp=amp)
    p_cal, y_cal = predict_proba(model, dl_calB, device, amp=amp)
    p_test, y_t  = predict_proba(model, dl_test,  device, amp=amp)
    assert np.array_equal(y_t, y_test2.astype(np.int64)), "Test labels mismatch after loader."

    meta = {
        "img_npz": img_npz,
        "model": args.model,
        "K": K,
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "calibA_frac": float(args.calibA_frac),
        "calib_split_seed": int(args.calib_split_seed),
        "calibA_size": int(X_calA.shape[0]),
        "calibB_size": int(X_calB.shape[0]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "freeze_backbone": bool(args.freeze_backbone),
        "note": "Keys: p_sel/y_sel, p_cal/y_cal, p_test/y_test",
        "val_idxA": idxA.tolist(),
        "val_idxB": idxB.tolist(),
        "val_idxT": idxT.tolist(),
    }

    # class counts in training pool
    counts_pool = np.bincount(y_train.astype(int), minlength=K).astype(np.int64)
    tail_frac = 0.2 #수정 가능. 일단 0.2로 맞춰둠
    m = int(np.ceil(tail_frac * K))
    tail_set = np.argsort(counts_pool)[:m].astype(np.int64)

    os.makedirs(os.path.dirname(out_prob_npz), exist_ok=True)
    np.savez_compressed(
        out_prob_npz,
        p_sel=p_sel.astype(np.float32), y_sel=y_sel.astype(np.int64),
        p_cal=p_cal.astype(np.float32), y_cal=y_cal.astype(np.int64),
        p_test=p_test.astype(np.float32), y_test=y_test2.astype(np.int64),
        counts_pool=counts_pool,
        tail_set = tail_set,
        meta=np.array([meta], dtype=object),
    )

    print(f"[saved probs npz] {out_prob_npz}")
    print("[shapes] p_sel", p_sel.shape, "p_cal", p_cal.shape, "p_test", p_test.shape)


if __name__ == "__main__":
    main()
