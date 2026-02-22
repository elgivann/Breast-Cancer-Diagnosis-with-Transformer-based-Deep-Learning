from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import auc, confusion_matrix, classification_report,roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, brier_score_loss
from torchvision.transforms import v2
from tqdm import tqdm
import colorama
colorama.just_fix_windows_console()
import pandas as pd
import numpy as np
import argparse
from PIL import Image
from datetime import datetime
from pathlib import Path
import time
import random
import warnings
import os
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


warnings.filterwarnings("ignore", message=".*epoch parameter.*")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

batch_size = 16
imageSize = 256
patchSize = 16
embeddingSize = 128
headCount=8
mlpSize=512 
dropRate=0.1
depth=8
inputChannels = 3

# dataset splitting
def splitByPatientKFold(filepath, folds=5, randomSeed=42):
    data = pd.read_csv(filepath)
    data["patientId"] = data["image"].apply(lambda x: Path(x).stem)

    patientIds = data["patientId"].unique()
    np.random.seed(randomSeed)
    np.random.shuffle(patientIds)

    kfold = KFold(n_splits=folds, shuffle=True, random_state=randomSeed)
    allFolds = []

    for foldNumber, (trainIndex, validationIndex) in enumerate(kfold.split(patientIds), start=1):
        trainPatients = patientIds[trainIndex]
        validationPatients = patientIds[validationIndex]

        trainIndices = data[data["patientId"].isin(trainPatients)].index.tolist()
        validationIndices = data[data["patientId"].isin(validationPatients)].index.tolist()

        # leakage check
        assert len(set(data.loc[trainIndices, "patientId"]) & set(data.loc[validationIndices, "patientId"])) == 0, "Data Leakage detected!"

        print(f"Fold {foldNumber}: Training patients {len(trainPatients)}, Validation patients {len(validationPatients)}")
        print(f"Fold {foldNumber}: Training images {len(trainIndices)}, Validation images {len(validationIndices)}")

        allFolds.append((trainIndices, validationIndices))

    return allFolds

# data augmentation
def augmentation(train=True, test=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        return v2.Compose([
            v2.Resize((imageSize, imageSize)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(15),
            v2.RandomAdjustSharpness(1.5, p=0.3),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=mean, std=std)
        ])
    elif test:
        return [v2.Compose([
                v2.Resize((imageSize, imageSize)),
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Normalize(mean=mean, std=std)
            ])
        ]
    else:
        return v2.Compose([
            v2.Resize((imageSize, imageSize)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=mean, std=std)
        ])

class CustomDataset(Dataset):
    def __init__(self, filepath, train=False, test=False):
        self.data = pd.read_csv(filepath)
        self.data = self.data[self.data["label"].isin([1, 2])].reset_index(drop=True)
        self.data["label"] = self.data["label"].apply(lambda x: 0 if x == 1 else 1)

        self.train = train
        self.test = test
        self.augment = augmentation(train=train, test=test)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data.loc[index, "image"]
        image = Image.open(img_path).convert("RGB")

        if self.test:
            return [transform(image) for transform in self.augment], torch.tensor(self.data.loc[index, "label"], dtype=torch.long)
        else:
            image = self.augment(image)
            label = self.data.loc[index, "label"]
            return image, torch.tensor(label, dtype=torch.long)

class PatchEmbedding(nn.Module):
    def __init__(self, imageSize=imageSize, patchSize=patchSize, inputChannels=inputChannels, embeddingSize=embeddingSize):
        super().__init__()
        self.numPatches = (imageSize // patchSize) ** 2
        self.projection = nn.Conv2d(inputChannels, embeddingSize, kernel_size=patchSize, stride=patchSize, padding=0)
        self.classToken = nn.Parameter(torch.zeros(1, 1, embeddingSize))
        self.positionEmbedding = nn.Parameter(torch.zeros(1, self.numPatches + 1, embeddingSize))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        classToken = self.classToken.expand(batch_size, -1, -1)
        x = torch.cat((classToken, x), dim=1)
        x = x + self.positionEmbedding
        return x

class MLPBlock(nn.Module):
    def __init__(self, embeddingSize=embeddingSize, headCount=headCount, mlpSize=mlpSize, dropRate=dropRate):
        super().__init__()
        self.norm1 = nn.LayerNorm(embeddingSize)
        self.attention = nn.MultiheadAttention(embed_dim=embeddingSize, num_heads=headCount, dropout=dropRate)
        self.norm2 = nn.LayerNorm(embeddingSize)
        self.mlp = nn.Sequential(
            nn.Linear(embeddingSize, mlpSize),
            nn.GELU(),
            nn.Dropout(dropRate),
            nn.Linear(mlpSize, embeddingSize),
            nn.Dropout(dropRate)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        x_t = x_norm.transpose(0, 1)
        attn_out, _ = self.attention(x_t, x_t, x_t)
        attn_out = attn_out.transpose(0, 1)
        x = x + attn_out
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, depth=depth, headCount=headCount, mlpSize=mlpSize):
        super().__init__()
        self.patchEmbed = PatchEmbedding()
        self.transformerBlocks = nn.ModuleList([MLPBlock(embeddingSize, headCount, mlpSize) for _ in range(depth)])
        self.finalNorm = nn.LayerNorm(embeddingSize)

    def forward(self, x):
        x = self.patchEmbed(x)
        for block in self.transformerBlocks:
            x = block(x)
        x = self.finalNorm(x)
        return x[:, 0, :]

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visionTransformer = VisionTransformer()
        self.classifier = nn.Sequential(
            nn.Linear(embeddingSize, embeddingSize),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embeddingSize, 2)
        )

    def forward(self, x):
        features = self.visionTransformer(x)
        return self.classifier(features)

def train(csv_filepath, model_filepath, epochs, device):
    print(f"Running training with {epochs} epochs")

    dataset = CustomDataset(csv_filepath, train=True)
    
    allFolds = splitByPatientKFold(csv_filepath)
    trainId, valId = allFolds[0] 
    print(f"Train samples: {len(trainId)}, Validation samples: {len(valId)}")

    train_data = DataLoader(Subset(dataset, trainId), batch_size=batch_size, shuffle=True)
    val_data = DataLoader(Subset(dataset, valId), batch_size=batch_size, shuffle=False)

    model = MyModel().to(device)
    labels = dataset.data["label"].values
    class_counts = np.bincount(labels)
    class_weights = torch.tensor([class_counts.sum() / class_counts[i] for i in range(2)],
    dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    base_lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-4)

    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters= 5)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - 10), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])

    best_val_acc = 0.0
    patience_counter = 0
    patience = 30
    
    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        for batch_idx, (images, labels) in enumerate(train_data):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            predicted = outputs.argmax(1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # for every 50 batches print the progress
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_data):
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(train_data)}] "
                      f"Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_data)
        train_acc = 100 * correct_train / total_train

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_data:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = outputs.argmax(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_filepath)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.8f}")

    print(f"Training finished. Best validation accuracy: {best_val_acc:.2f}%")

# testing on external dataset BUS-BRA
def execute(csv_filepath, model_filepath, out_filepath, device, tta_runs=10):
    dataset = CustomDataset(csv_filepath, test=True)
    os.makedirs('plot_dir', exist_ok=True)
    data = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Test samples: {len(dataset)}")

    model = MyModel().to(device)
    checkpoint = torch.load(model_filepath, map_location=device)
    model_dict = model.state_dict()

    filtered_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)
    missing_keys = set(model.state_dict().keys()) - set(filtered_dict.keys())
    if missing_keys:
        print(f"Warning: {len(missing_keys)} keys not loaded from checkpoint")

    run_name = Path(model_filepath).stem  # e.g. model_50
    plot_dir = Path("plots") / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    correct, total = 0, 0
    y_true, y_pred = [], []
    y_probs = []   # store for malignant probabilities

    with torch.no_grad():
        for idx, (images_list, label) in enumerate(data):
            tta_preds = []
            tta_runs_safe = min(tta_runs, len(images_list))
            for img in images_list[:tta_runs_safe]:
                img = img.to(device)
                tta_preds.append(F.softmax(model(img), dim=1))
            avg_pred = torch.mean(torch.stack(tta_preds), dim=0)
            malignant_prob = avg_pred[0, 1].item()
            y_probs.append(malignant_prob)
            y_true.append(label.item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)                
    y_probs_clipped = np.clip(y_probs, 1e-8, 1-1e-8)

    # ROC & PR curves per class
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    fpr_m, tpr_m, thresholds = roc_curve(y_true, y_probs)
    auc_m = auc(fpr_m, tpr_m)
    fpr_b, tpr_b, _ = roc_curve(1 - y_true, 1 - y_probs)
    auc_b = auc(fpr_b, tpr_b)

    youden_index = tpr_m - fpr_m
    best_threshold = thresholds[np.argmax(youden_index)]

    y_pred = (y_probs >= best_threshold).astype(int)

    test_acc = 100 * np.mean(y_pred == y_true)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Best threshold (Youden): {best_threshold:.4f}")

    test_acc = 100 * np.mean(y_pred == y_true)
    print(f"Test Accuracy: {test_acc:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=["benign", "malignant"],
        digits=4
    )
    print("\nClassification Report:\n", report)

    # compute metrics
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    # metrics for the malignant class
    sensitivity = TP[1] / (TP[1] + FN[1])   
    specificity = TN[1] / (TN[1] + FP[1])
    precision = TP[1] / (TP[1] + FP[1])
    f1 = 2 * precision * sensitivity / (precision + sensitivity)
    fppi = FP.sum() / len(y_true) 

    precision_m, recall_m, _ = precision_recall_curve(y_true, y_probs)
    precision_b, recall_b, _ = precision_recall_curve(1 - y_true, 1 - y_probs)
    pr_auc_m = auc(recall_m, precision_m)
    pr_auc_b = auc(recall_b, precision_b)

    plt.figure(figsize=(6,6))
    plt.plot(fpr_m, tpr_m, label=f'Malignant (AUC={auc_m:.3f})')
    plt.plot(fpr_b, tpr_b, label=f'Benign (AUC={auc_b:.3f})')
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot_dir/roc_curves.png')
    plt.close()

    plt.figure(figsize=(6,6))
    plt.plot(recall_m, precision_m, label=f'Malignant (PR AUC={pr_auc_m:.3f})')
    plt.plot(recall_b, precision_b, label=f'Benign (PR AUC={pr_auc_b:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot_dir/pr_curves.png')
    plt.close()

    # Calibration
    prob_true, prob_pred = calibration_curve(y_true.astype(int), y_probs_clipped, n_bins=10, strategy='uniform')
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model calibration')
    plt.plot([0,1],[0,1], linestyle='--', color='gray', label='Perfect calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot_dir/calibration_curve.png')
    plt.close()

    # Threshold sensitivity plot
    thresholds_grid = np.linspace(0,1,50)
    sens_list, spec_list = [], []
    for t in thresholds_grid:
        preds = (y_probs >= t).astype(int)
        cm_t = confusion_matrix(y_true, preds)
        if cm_t.shape == (2,2):
            tn, fp, fn, tp = cm_t.ravel()
        else:
            tn, fp, fn, tp = 0,0,0,0
        sens_list.append(tp / (tp + fn))
        spec_list.append(tn / (tn + fp))

    plt.figure(figsize=(6,6))
    plt.plot(thresholds_grid, sens_list, label='Sensitivity')
    plt.plot(thresholds_grid, spec_list, label='Specificity')
    plt.xlabel('Threshold')
    plt.ylabel('Value')
    plt.title('Threshold Sensitivity Analysis')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot_dir/threshold_sensitivity.png')
    plt.close()

    # ECE
    def compute_ece(probs, labels, bins=10):
        bin_edges = np.linspace(0, 1, bins + 1)
        ece = 0
        for i in range(bins):
            mask = (probs > bin_edges[i]) & (probs <= bin_edges[i+1])
            if np.sum(mask) > 0:
                acc = np.mean(labels[mask] == (probs[mask] > 0.5))
                conf = np.mean(probs[mask])
                ece += np.abs(acc - conf) * np.sum(mask) / len(probs)
        return ece

    ece = compute_ece(y_probs_clipped, y_true)
    brier = brier_score_loss(y_true, y_probs_clipped)

    # Bootstrap CI
    def bootstrap_auc(labels, probs, n_boot=1000):
        rng = np.random.RandomState(42)
        scores = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(probs), len(probs))
            if len(np.unique(labels[idx])) < 2:
                continue
            scores.append(roc_auc_score(labels[idx], probs[idx]))
        return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

    ci_lower, ci_upper = bootstrap_auc(y_true, y_probs)

    # Print summary
    print(f"AUC Malignant: {auc_m:.4f}, Benign: {auc_b:.4f}")
    print(f"PR AUC Malignant: {pr_auc_m:.4f}, Benign: {pr_auc_b:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}, F1-score: {f1:.4f}, FPPI: {fppi:.4f}")
    print(f"AUC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"ECE: {ece:.4f}, Brier Score: {brier:.4f}")

    # Save outputs
    dataset.data.to_csv(out_filepath, index=False)
    np.save("test_probs.npy", y_probs)
    np.save("test_labels.npy", y_true)
    print(f"Saved predictions to {out_filepath} and plots as PNGs")
    
def canOverwrite(filepath):
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return True

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running model on {device} device")

    csv_filepath = "dataBUS-CoT.csv"
    out_filepath = "out.csv"
    model_filepath = f"models/model_{datetime.now():%d-%m-%Y_%H-%M}.pth"

    parser = argparse.ArgumentParser(description="Transformer-based AI for breast cancer detection (benign vs malignant).")
    parser.add_argument("filepath", help="Input CSV filepath.")
    parser.add_argument("-m", "--model", help="Model filepath (save/load).")
    parser.add_argument("-e", "--epochs", type=int, help="Training epochs.")
    parser.add_argument("-o", "--out", help="Output CSV filepath.")
    parser.add_argument("-t", "--train", action="store_true", help="Enable training mode.")
    parser.add_argument("--tta", type=int, default=5, help="Test-time augmentations (default: 5)")
    args = parser.parse_args()

    if args.model:
        model_filepath = args.model
    if args.epochs:
        epochs = args.epochs
    else:
        epochs = 50
    if args.out:
        out_filepath = args.out

    csv_filepath = args.filepath

    if args.train:
        if not canOverwrite(model_filepath):
            print("Cannot save model file.")
            exit(1)
        train(csv_filepath, model_filepath, epochs, device)
    else:
        if not args.model:
            print("Must provide model file for testing (-m <path>)")
            exit(2)
        if not canOverwrite(out_filepath):
            print("Cannot save output file.")
            exit(3)
        execute(csv_filepath, model_filepath, out_filepath, device, args.tta)

if __name__ == "__main__":
    main()
