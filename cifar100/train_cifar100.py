import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_splits(n_train=30000, n_select=10000, n_calib=10000, seed=1):
    # CIFAR-100 train set size = 50000
    assert n_train + n_select + n_calib <= 50000
    rng = np.random.default_rng(seed)
    idx = rng.permutation(50000)
    tr = idx[:n_train]
    sel = idx[n_train:n_train+n_select]
    cal = idx[n_train+n_select:n_train+n_select+n_calib]
    return tr, sel, cal

@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs_list = []
    y_list = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs_list.append(p)
        y_list.append(y.numpy())
    P = np.concatenate(probs_list, axis=0)
    Y = np.concatenate(y_list, axis=0)
    return P, Y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./out/cifar100_probs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_gpu", action="store_true", help="use CUDA if available")
    parser.add_argument("--n_train", type=int, default=30000)
    parser.add_argument("--n_select", type=int, default=10000)
    parser.add_argument("--n_calib", type=int, default=10000)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # CIFAR-100 normalization commonly used for CIFAR training
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # Base datasets
    full_train = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=True, transform=train_tf
    )
    full_train_eval = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=False, transform=eval_tf
    )
    testset = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=eval_tf
    )

    # Indices for train/select/calib split (from 50k train)
    idx_tr, idx_sel, idx_cal = make_splits(
        n_train=args.n_train, n_select=args.n_select, n_calib=args.n_calib, seed=args.seed
    )

    ds_tr  = Subset(full_train, idx_tr)          # training augmentation
    ds_sel = Subset(full_train_eval, idx_sel)    # evaluation transform
    ds_cal = Subset(full_train_eval, idx_cal)    # evaluation transform

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    dl_sel = DataLoader(ds_sel, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    dl_cal = DataLoader(ds_cal, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type=="cuda"))
    dl_te  = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type=="cuda"))

    # Model: ResNet18 adjusted for CIFAR (simple approach)
    model = torchvision.models.resnet18(weights=None)
    # CIFAR: input 32x32, best practice is modify first conv+remove maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max(args.epochs//2,1)], gamma=0.1)

    # Quick training
    model.train()
    for ep in range(1, args.epochs + 1):
        total = 0
        correct = 0
        running_loss = 0.0
        for x, y in dl_tr:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            total += x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()

        scheduler.step()
        print(f"[epoch {ep}] loss={running_loss/total:.4f} acc={correct/total:.4f}")

    # Predict probabilities on selection/calib/test
    p_sel, y_sel = predict_proba(model, dl_sel, device)
    p_cal, y_cal = predict_proba(model, dl_cal, device)
    p_tst, y_tst = predict_proba(model, dl_te,  device)

    # Save as .npz (easy to load in R via reticulate or small parser)
    out_path = os.path.join(
        args.out_dir,
        f"cifar100_probs_seed{args.seed}_e{args.epochs}_tr{args.n_train}_sel{args.n_select}_cal{args.n_calib}.npz"
    )
    np.savez_compressed(
        out_path,
        p_sel=p_sel, y_sel=y_sel,
        p_cal=p_cal, y_cal=y_cal,
        p_tst=p_tst, y_tst=y_tst,
        idx_tr=idx_tr, idx_sel=idx_sel, idx_cal=idx_cal
    )
    print(f"[saved] {out_path}")
    print("[shapes]",
          "p_sel", p_sel.shape,
          "p_cal", p_cal.shape,
          "p_tst", p_tst.shape)

if __name__ == "__main__":
    main()
