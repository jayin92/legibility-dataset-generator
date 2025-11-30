import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoProcessor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
import config


# --- Dataset ---
class LegibilityDataset(Dataset):
    def __init__(self, jsonl_file, processor, transform=None):
        self.data = []
        self.processor = processor
        self.transform = transform
        
        # Label Mapping
        self.label_map = {
            "a_much_better": 1.0,
            "a_better": 0.75,
            "equal": 0.5,
            "b_better": 0.25,
            "b_much_better": 0.0,
            # Fallback for legacy labels if any
            "a": 0.75,
            "b": 0.25
        }

        print(f"Loading data from {jsonl_file}...")
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if 'choice' in item and item['choice'] in self.label_map:
                        self.data.append(item)
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(self.data)} pairs.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        image_a_path = os.path.join(config.OUTPUT_DIR, item['image_a'])
        image_b_path = os.path.join(config.OUTPUT_DIR, item['image_b'])
        
        try:
            image_a = Image.open(image_a_path).convert("RGB")
            image_b = Image.open(image_b_path).convert("RGB")
        except Exception as e:
            # Handle missing images gracefully
            # print(f"Error loading image: {e}")
            image_a = Image.new("RGB", (512, 512))
            image_b = Image.new("RGB", (512, 512))

        # Process images for SigLIP
        inputs_a = self.processor(images=image_a, return_tensors="pt")
        inputs_b = self.processor(images=image_b, return_tensors="pt")
        
        pixel_values_a = inputs_a['pixel_values'].squeeze(0)
        pixel_values_b = inputs_b['pixel_values'].squeeze(0)
        
        label = self.label_map.get(item['choice'], 0.5)
        
        return pixel_values_a, pixel_values_b, torch.tensor(label, dtype=torch.float32)

# --- Model Components ---

class SigLIPFeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super(SigLIPFeatureExtractor, self).__init__()
        print(f"Loading SigLIP model: {model_name}...")
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, pixel_values):
        # SigLIP vision model output
        outputs = self.model.vision_model(pixel_values=pixel_values)
        return outputs.pooler_output

class RankNetScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.2):
        super(RankNetScorer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid() # Ensure output is in [0, 1]
        )

    def forward(self, x):
        return self.net(x)

class RankNetLoss(nn.Module):
    def __init__(self, scaling_factor):
        super(RankNetLoss, self).__init__()
        self.scaling_factor = scaling_factor
        self.bce_loss = nn.BCELoss()

    def forward(self, score_i, score_j, target_p_ij):
        # score_i, score_j are in [0, 1]
        score_diff = score_i - score_j
        
        # Scale the difference to allow sigmoid to reach 0 and 1
        scaled_diff = self.scaling_factor * score_diff
        
        # Predicted probability that i > j
        pred_p_ij = torch.sigmoid(scaled_diff)
        
        # BCE Loss
        loss = self.bce_loss(pred_p_ij, target_p_ij.unsqueeze(1))
        return loss

class LegibilityPredictor(nn.Module):
    def __init__(self, feature_extractor, scorer):
        super(LegibilityPredictor, self).__init__()
        self.feature_extractor = feature_extractor
        self.scorer = scorer

    def forward(self, pixel_values_i, pixel_values_j=None):
        # If only one image provided, return score
        features_i = self.feature_extractor(pixel_values_i)
        score_i = self.scorer(features_i)
        
        if pixel_values_j is None:
            return score_i
            
        features_j = self.feature_extractor(pixel_values_j)
        score_j = self.scorer(features_j)
        
        return score_i, score_j

# --- Training Loop ---

def train(args):
    # Initialize WandB
    wandb.init(project="legibility-predictor", config=vars(args), name=os.path.splitext(os.path.basename(args.input_file))[0])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup Output Directory
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_dir = os.path.join("outputs", base_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving checkpoints to: {output_dir}")

    # 1. Prepare Data
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    # Split JSONL into train/val
    all_data = []
    with open(args.input_file, 'r') as f:
        all_data = f.readlines()
    
    train_lines, val_lines = train_test_split(all_data, test_size=0.2, random_state=42)
    
    with open("train_temp.jsonl", "w") as f:
        f.writelines(train_lines)
    with open("val_temp.jsonl", "w") as f:
        f.writelines(val_lines)
        
    train_dataset = LegibilityDataset("train_temp.jsonl", processor)
    val_dataset = LegibilityDataset("val_temp.jsonl", processor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # 2. Initialize Model
    feature_extractor = SigLIPFeatureExtractor(model_name=args.model_name)
    # Get embedding dimension dynamically
    dummy_input = torch.randn(1, 3, 512, 512) # Standard SigLIP2 input size
    with torch.no_grad():
        dummy_out = feature_extractor.model.vision_model(dummy_input).pooler_output
        input_dim = dummy_out.shape[1]
    
    print(f"Feature dimension: {input_dim}")
    
    scorer = RankNetScorer(input_dim=input_dim)
    model = LegibilityPredictor(feature_extractor, scorer).to(device)
    
    criterion = RankNetLoss(scaling_factor=args.scaling_factor)
    optimizer = optim.Adam(model.scorer.parameters(), lr=args.learning_rate) # Only train scorer

    # 3. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for pixel_values_a, pixel_values_b, labels in progress_bar:
            pixel_values_a = pixel_values_a.to(device, non_blocking=True)
            pixel_values_b = pixel_values_b.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            score_a, score_b = model(pixel_values_a, pixel_values_b)
            loss = criterion(score_a, score_b, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Log batch loss
            wandb.log({"batch_train_loss": loss.item(), "epoch": epoch + 1})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_pairs = 0
        total_pairs = 0
        
        with torch.no_grad():
            for pixel_values_a, pixel_values_b, labels in val_loader:
                pixel_values_a = pixel_values_a.to(device, non_blocking=True)
                pixel_values_b = pixel_values_b.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                score_a, score_b = model(pixel_values_a, pixel_values_b)
                loss = criterion(score_a, score_b, labels)
                val_loss += loss.item()
                
                diff = score_a - score_b
                preds = (diff > 0).float()
                
                non_ties = (labels != 0.5)
                if non_ties.sum() > 0:
                    binary_preds = preds[non_ties]
                    binary_targets = (labels[non_ties] > 0.5).float().unsqueeze(1)
                    correct_pairs += (binary_preds == binary_targets).sum().item()
                    total_pairs += non_ties.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_pairs / total_pairs if total_pairs > 0 else 0.0
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc (non-ties): {val_acc:.4f}")
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        })
        
        # Save Latest Checkpoint
        latest_path = os.path.join(output_dir, "checkpoint_latest.pth")
        torch.save(model.state_dict(), latest_path)
        
        # Save Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(output_dir, "checkpoint_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")
            wandb.log({"best_val_loss": best_val_loss})

    # Cleanup temp files
    if os.path.exists("train_temp.jsonl"): os.remove("train_temp.jsonl")
    if os.path.exists("val_temp.jsonl"): os.remove("val_temp.jsonl")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the JSONL file with ratings.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--model_name", type=str, default="google/siglip2-so400m-patch16-512", help="HuggingFace model name for SigLIP.")
    parser.add_argument("--scaling_factor", type=float, default=10.0, help="Scaling factor for RankNet loss.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    args = parser.parse_args()
    
    train(args)
