import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoProcessor, VisionEncoderDecoderModel, TrOCRProcessor
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from tqdm import tqdm
import wandb

import config


# --- Reproducibility ---

def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Dataset ---

class LegibilityDataset(Dataset):
    """Dataset for pairwise legibility comparisons."""
    
    LABEL_MAP = {
        "a_much_better": 1.0,
        "a_better": 0.75,
        "equal": 0.5,
        "b_better": 0.25,
        "b_much_better": 0.0,
        # Legacy labels
        "a": 0.75,
        "b": 0.25,
    }

    def __init__(self, jsonl_file: str, processor, image_dir: str = None):
        self.processor = processor
        self.image_dir = image_dir or config.OUTPUT_DIR
        self.data = self._load_data(jsonl_file)

    def _load_data(self, jsonl_file: str) -> list:
        """Load and validate data from JSONL file."""
        data = []
        invalid_choices = set()
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    choice = item.get('choice')
                    if isinstance(choice, str):
                        # Strip quotes and whitespace
                        choice = choice.strip().strip('"').strip("'")
                        
                    if choice in self.LABEL_MAP:
                        # Update the item with the cleaned choice
                        item['choice'] = choice
                        data.append(item)
                    else:
                        invalid_choices.add(item.get('choice'))
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(data)} valid pairs from {jsonl_file}")
        if invalid_choices:
            print(f"Ignored choices: {invalid_choices}")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def _load_image(self, path: str) -> Image.Image:
        """Load image with error handling."""
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", (512, 512), color=(128, 128, 128))

    def __getitem__(self, idx: int):
        item = self.data[idx]
        
        image_a = self._load_image(os.path.join(self.image_dir, item['image_a']))
        image_b = self._load_image(os.path.join(self.image_dir, item['image_b']))

        inputs_a = self.processor(images=image_a, return_tensors="pt")
        inputs_b = self.processor(images=image_b, return_tensors="pt")
        
        pixel_values_a = inputs_a['pixel_values'].squeeze(0)
        pixel_values_b = inputs_b['pixel_values'].squeeze(0)
        
        label = self.LABEL_MAP[item['choice']]
        
        return pixel_values_a, pixel_values_b, torch.tensor(label, dtype=torch.float32)


# --- Model Components ---

class SigLIPFeatureExtractor(nn.Module):
    """Frozen SigLIP vision encoder for feature extraction."""
    
    def __init__(self, model_name: str):
        super().__init__()
        print(f"Loading SigLIP model: {model_name}...")
        self.model = AutoModel.from_pretrained(model_name)
        self._freeze_parameters()
        
    def _freeze_parameters(self):
        """Freeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
            
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values=pixel_values)
        return outputs.pooler_output
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the feature extractor."""
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            output = self.model.vision_model(dummy_input).pooler_output
        return output.shape[1]


class TrOCRFeatureExtractor(nn.Module):
    """Frozen TrOCR encoder for feature extraction."""
    
    def __init__(self, model_name: str):
        super().__init__()
        print(f"Loading TrOCR model: {model_name}...")
        # TrOCR is an encoder-decoder, we only need the encoder
        full_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.encoder = full_model.encoder
        self._freeze_parameters()
        
    def _freeze_parameters(self):
        """Freeze all parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.encoder(pixel_values=pixel_values)
            # Use the CLS token (first token) from the last hidden state
            # Shape: [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
            last_hidden_state = outputs.last_hidden_state
            cls_token = last_hidden_state[:, 0, :]
        return cls_token

    def get_output_dim(self) -> int:
        """Get the output dimension of the feature extractor."""
        # TrOCR base usually expects 384x384, but processor handles resizing.
        # We'll use a dummy input to check.
        # Note: TrOCR image size depends on the specific model config.
        # We'll assume the processor has resized it correctly before this.
        # Standard ViT input is often 224 or 384.
        # Let's try to infer from config or just run a dummy.
        dummy_input = torch.randn(1, 3, 384, 384) 
        with torch.no_grad():
            output = self.encoder(dummy_input).last_hidden_state[:, 0, :]
        return output.shape[1]


class RankNetScorer(nn.Module):
    """MLP scorer that maps features to a legibility score in [0, 1]."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RankNetLoss(nn.Module):
    """RankNet loss for pairwise ranking."""
    
    def __init__(self, scaling_factor: float = 10.0):
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(
        self, 
        score_i: torch.Tensor, 
        score_j: torch.Tensor, 
        target_p_ij: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RankNet loss.
        
        Args:
            score_i: Scores for item i, shape [B, 1]
            score_j: Scores for item j, shape [B, 1]
            target_p_ij: Target probability that i > j, shape [B]
        """
        score_diff = score_i - score_j
        scaled_diff = self.scaling_factor * score_diff
        
        # Use BCEWithLogitsLoss for numerical stability and AMP safety
        # scaled_diff acts as the logits
        target = target_p_ij.view_as(scaled_diff)
        loss = nn.functional.binary_cross_entropy_with_logits(scaled_diff, target)
        return loss


class LegibilityPredictor(nn.Module):
    """Full model combining feature extractor and scorer."""
    
    def __init__(self, feature_extractor: SigLIPFeatureExtractor, scorer: RankNetScorer):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.scorer = scorer

    def forward(
        self, 
        pixel_values_i: torch.Tensor, 
        pixel_values_j: torch.Tensor = None
    ):
        features_i = self.feature_extractor(pixel_values_i)
        score_i = self.scorer(features_i)
        
        if pixel_values_j is None:
            return score_i
            
        features_j = self.feature_extractor(pixel_values_j)
        score_j = self.scorer(features_j)
        
        return score_i, score_j
    
    def score(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get legibility score for a single image."""
        return self.forward(pixel_values)


# --- Training Utilities ---

class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def create_dataloaders(args, processor) -> tuple:
    """Create train and validation dataloaders."""
    # Load and split data
    with open(args.input_file, 'r') as f:
        all_lines = f.readlines()
    
    train_lines, val_lines = train_test_split(
        all_lines, 
        test_size=args.val_split, 
        random_state=args.seed
    )
    
    # Create temp files
    train_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    val_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
    
    train_file.writelines(train_lines)
    val_file.writelines(val_lines)
    train_file.close()
    val_file.close()
    
    # Create datasets
    train_dataset = LegibilityDataset(train_file.name, processor)
    val_dataset = LegibilityDataset(val_file.name, processor)
    
    # Cleanup temp files
    os.unlink(train_file.name)
    os.unlink(val_file.name)
    
    # Create dataloaders
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'persistent_workers': args.num_workers > 0,
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace
):
    """Save full checkpoint for resuming training."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'args': vars(args),
    }, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer = None, scheduler = None):
    """Load checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_val_loss', float('inf'))


# --- Validation ---

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True
) -> dict:
    """Run validation and compute metrics."""
    model.eval()
    
    total_loss = 0.0
    all_score_diffs = []
    all_labels = []
    correct_pairs = 0
    total_non_ties = 0
    
    for pixel_values_a, pixel_values_b, labels in val_loader:
        pixel_values_a = pixel_values_a.to(device, non_blocking=True)
        pixel_values_b = pixel_values_b.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            score_a, score_b = model(pixel_values_a, pixel_values_b)
            loss = criterion(score_a, score_b, labels)
        
        total_loss += loss.item()
        
        # Collect for correlation
        score_diff = (score_a - score_b).squeeze().cpu().numpy()
        all_score_diffs.extend(score_diff.tolist() if hasattr(score_diff, 'tolist') else [score_diff])
        all_labels.extend(labels.cpu().numpy().tolist())
        
        # Accuracy on non-ties
        preds = (score_a > score_b).float().squeeze()
        non_ties = labels != 0.5
        
        if non_ties.sum() > 0:
            targets = (labels[non_ties] > 0.5).float()
            correct_pairs += (preds[non_ties] == targets).sum().item()
            total_non_ties += non_ties.sum().item()
    
    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_pairs / total_non_ties if total_non_ties > 0 else 0.0
    
    # Spearman correlation
    spearman_corr, _ = spearmanr(all_score_diffs, all_labels)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'spearman': spearman_corr if not np.isnan(spearman_corr) else 0.0,
    }


# --- Training Loop ---

def train(args):
    """Main training function."""
    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Output directory
    model_name_safe = args.model_name.replace("/", "-")
    base_name = f"{Path(args.input_file).stem}_{model_name_safe}"
    output_dir = Path("outputs") / base_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to: {output_dir}")
    
    # Initialize WandB
    wandb.init(
        project="legibility-predictor",
        config=vars(args),
        name=base_name,
        resume="allow" if args.resume else None
    )
    
    # Data
    if "trocr" in args.model_name.lower():
        processor = TrOCRProcessor.from_pretrained(args.model_name)
    else:
        processor = AutoProcessor.from_pretrained(args.model_name)
        
    train_loader, val_loader = create_dataloaders(args, processor)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    if "trocr" in args.model_name.lower():
        feature_extractor = TrOCRFeatureExtractor(args.model_name)
    else:
        feature_extractor = SigLIPFeatureExtractor(args.model_name)
        
    input_dim = feature_extractor.get_output_dim()
    print(f"Feature dimension: {input_dim}")
    
    scorer = RankNetScorer(input_dim=input_dim, hidden_dim=args.hidden_dim)
    model = LegibilityPredictor(feature_extractor, scorer).to(device)
    
    # Optionally compile model (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Loss, optimizer, scheduler
    criterion = RankNetLoss(scaling_factor=args.scaling_factor)
    optimizer = optim.AdamW(
        model.scorer.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        resume_path = output_dir / "checkpoint_latest.pth"
        if resume_path.exists():
            print(f"Resuming from {resume_path}")
            start_epoch, best_val_loss = load_checkpoint(
                resume_path, model, optimizer, scheduler
            )
            start_epoch += 1
            print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        
        train_loss = 0.0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            leave=True
        )
        
        for step, (pixel_values_a, pixel_values_b, labels) in enumerate(progress_bar):
            pixel_values_a = pixel_values_a.to(device, non_blocking=True)
            pixel_values_b = pixel_values_b.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass with AMP
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                score_a, score_b = model(pixel_values_a, pixel_values_b)
                loss = criterion(score_a, score_b, labels)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.scorer.parameters(),
                    args.grad_clip
                )
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log batch metrics
            if step % args.log_interval == 0:
                wandb.log({
                    "batch/train_loss": loss.item(),
                    "batch/learning_rate": optimizer.param_groups[0]['lr'],
                    "batch/step": epoch * len(train_loader) + step,
                })
        
        
        # Step scheduler
        scheduler.step()
        
        # Validation
        val_metrics = validate(model, val_loader, criterion, device, args.use_amp)
        avg_train_loss = train_loss / len(train_loader)
        
        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Spearman: {val_metrics['spearman']:.4f}"
        )
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "val/loss": val_metrics['loss'],
            "val/accuracy": val_metrics['accuracy'],
            "val/spearman": val_metrics['spearman'],
            "learning_rate": optimizer.param_groups[0]['lr'],
        })
        
        # Save latest checkpoint
        save_checkpoint(
            output_dir / "checkpoint_latest.pth",
            model, optimizer, scheduler,
            epoch, best_val_loss, args
        )
        
        # Save best checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                output_dir / "checkpoint_best.pth",
                model, optimizer, scheduler,
                epoch, best_val_loss, args
            )
            print(f"  → Saved best model (val_loss: {best_val_loss:.4f})")
            wandb.log({"best_val_loss": best_val_loss})
        
        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Save final model (scorer only for inference)
    torch.save(model.scorer.state_dict(), output_dir / "scorer_final.pth")
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    
    wandb.finish()


# --- Entry Point ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a legibility predictor using RankNet"
    )
    
    # Data
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to JSONL file with pairwise ratings")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    
    # Model
    parser.add_argument("--model_name", type=str,
                        default="google/siglip2-so400m-patch16-512",
                        help="HuggingFace model name for SigLIP")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for scorer MLP")
    parser.add_argument("--scaling_factor", type=float, default=10.0,
                        help="Scaling factor for RankNet loss")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay for AdamW")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm (0 to disable)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    
    # Performance
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (PyTorch 2.0+)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Batch logging interval")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)