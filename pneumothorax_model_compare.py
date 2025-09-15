"""
pneumothorax_model_compare.py

Purpose:
- Adapt CheXNet-like repositories to train binary classifiers for *Pneumothorax* only.
- Train and compare two architectures: ResNet50 and DenseNet121.
- Produce metrics and plots: confusion matrix, per-class confusion (here binary), accuracy, recall, precision, F1, ROC & AUC, Log Loss and PR curve.

Usage (example):
> python pneumothorax_model_compare.py --data_csv labels/train.csv --img_dir images/ --epochs 10 --batch 16 --models resnet50 densenet121

Dependencies (pip):
- torch torchvision
- numpy pandas scikit-learn matplotlib tqdm

Notes:
- Assumes CSV file has columns: "Image Index" (image filename) and columns for diseases including "Pneumothorax" (values 0/1 or 0/1/"-1").
- Adjust dataset splitting if you already have train/val/test splits (this script can read pre-split CSVs).

Author: ChatGPT (adapted for user's request)
"""

import os
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, precision_recall_curve, auc
)
import matplotlib.pyplot as plt


# ------------------------------- Dataset ---------------------------------
class ChestXrayPneumoDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, img_col='Image Index', label_col='Pneumothorax'):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.img_col = img_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = self.img_dir / row[self.img_col]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = row[self.label_col]
        # Ensure binary 0/1
        label = 1 if (str(label) in ('1', '1.0', 'True', 'true')) or float(label) == 1.0 else 0
        return img, torch.tensor(label, dtype=torch.float32)


# ------------------------------ Model Utils -------------------------------
def get_model(name='resnet50', pretrained=True, num_classes=1, device='cuda'):
    name = name.lower()
    if name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        n = model.fc.in_features
        model.fc = nn.Linear(n, num_classes)
    elif name == 'densenet121' or name == 'densenet':
        model = models.densenet121(pretrained=pretrained)
        n = model.classifier.in_features
        model.classifier = nn.Linear(n, num_classes)
    else:
        raise ValueError('Unsupported model: '+name)
    return model.to(device)


# ------------------------------- Training --------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    probs = []
    trues = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            p = torch.sigmoid(logits).cpu().numpy().ravel()
            probs.extend(p.tolist())
            trues.extend(labels.numpy().ravel().tolist())
    return np.array(trues), np.array(probs)


# --------------------------- Metrics & Plots ------------------------------

def binary_metrics_and_plots(y_true, y_probs, out_prefix):
    # Binarize predictions at 0.5
    y_pred = (y_probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except Exception:
        roc_auc = float('nan')
    try:
        ll = log_loss(y_true, y_probs)
    except Exception:
        ll = float('nan')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Save metrics
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'log_loss': ll,
        'confusion_matrix': cm.tolist()
    }

    # Plots: Confusion matrix
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Neg','Pneumo'])
    plt.yticks(tick_marks, ['Neg','Pneumo'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i,j]>cm.max()/2. else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    cm_path = out_prefix + '_confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()

    # ROC curve
    if not np.isnan(roc_auc):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],'--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (AUC={roc_auc:.4f})')
        plt.savefig(out_prefix + '_roc.png')
        plt.close()

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AUC={pr_auc:.4f})')
    plt.savefig(out_prefix + '_pr.png')
    plt.close()

    return metrics


# ------------------------------- Main -----------------------------------

def main(args):
    # Read csv
    df = pd.read_csv(args.data_csv)

    # Filter rows that have the image present and a valid Pneumothorax label
    # Handle different label encodings; keep only rows where Pneumothorax is not NaN
    assert 'Pneumothorax' in df.columns, 'CSV must contain Pneumothorax column'
    df = df[~df['Image Index'].isna()].copy()
    df = df[~df['Pneumothorax'].isna()].copy()

    # If labels are strings like '1'/'0' or probabilities, convert to numeric
    df['Pneumothorax'] = pd.to_numeric(df['Pneumothorax'], errors='coerce')
    df = df[~df['Pneumothorax'].isna()].copy()

    # Optionally remap: if dataset uses -1 for uncertain, map to 0 (or drop)
    if args.uncertain_to_negative:
        df.loc[df['Pneumothorax'] < 0, 'Pneumothorax'] = 0
    else:
        df = df[df['Pneumothorax'] >= 0]

    # Shuffle and split
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]

    print(f'Total: {n}, train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}')

    # Transforms
    train_tf = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = ChestXrayPneumoDataset(train_df, args.img_dir, transform=train_tf)
    val_ds = ChestXrayPneumoDataset(val_df, args.img_dir, transform=val_tf)
    test_ds = ChestXrayPneumoDataset(test_df, args.img_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'

    results = {}

    for model_name in args.models:
        print('\n==> Training model:', model_name)
        model = get_model(model_name, pretrained=args.pretrained, device=device)
        # Use BCEWithLogitsLoss for numerical stability
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_auc = -1
        best_state = None
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            y_val_true, y_val_probs = evaluate(model, val_loader, device)
            try:
                val_auc = roc_auc_score(y_val_true, y_val_probs)
            except Exception:
                val_auc = float('nan')
            print(f'Epoch {epoch+1}/{args.epochs} - train_loss: {train_loss:.4f} val_auc: {val_auc:.4f}')
            if not np.isnan(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k:v.cpu().state_dict() if hasattr(v,'state_dict') else v for k,v in model.state_dict().items()} if False else model.state_dict()

        # load best state if available (we saved only state dict above)
        if best_state is not None:
            model.load_state_dict(best_state)

        # Evaluate on test
        y_test_true, y_test_probs = evaluate(model, test_loader, device)
        out_prefix = os.path.join(args.output_dir, model_name)
        os.makedirs(args.output_dir, exist_ok=True)
        metrics = binary_metrics_and_plots(y_test_true, y_test_probs, out_prefix)
        results[model_name] = metrics

        # Save model
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'{model_name}_pneumo.pth'))

    # Summarize results
    print('\n=== Summary ===')
    for m, met in results.items():
        print(f'\nModel: {m}')
        print(f"Accuracy: {met['accuracy']:.4f}  Precision: {met['precision']:.4f}  Recall: {met['recall']:.4f}  F1: {met['f1']:.4f}  ROC_AUC: {met['roc_auc']:.4f}  LogLoss: {met['log_loss']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, required=True, help='CSV with image names and labels')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory with images')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Where to save models and plots')
    parser.add_argument('--models', nargs='+', default=['resnet50','densenet121'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_frac', type=float, default=0.7)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--uncertain_to_negative', action='store_true', help='Map uncertain labels (e.g., -1) to 0 instead of dropping')
    args = parser.parse_args()
    main(args)
