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
import config

# --- Configuration ---
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
RANKNET_SCALING_FACTOR = 10.0

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
            # Handle missing images gracefully (though ideally dataset should be clean)
            # Return a dummy item or handle in collate_fn. 
            # For simplicity, we'll just create a blank image (this is rare if data is consistent)
            print(f"Error loading image: {e}")
            image_a = Image.new("RGB", (224, 224))
            image_b = Image.new("RGB", (224, 224))

        # Process images for SigLIP
        # We process them individually. Processor returns dict with 'pixel_values'.
        inputs_a = self.processor(images=image_a, return_tensors="pt")
        inputs_b = self.processor(images=image_b, return_tensors="pt")
        
        pixel_values_a = inputs_a['pixel_values'].squeeze(0)
        pixel_values_b = inputs_b['pixel_values'].squeeze(0)
        
        label = self.label_map.get(item['choice'], 0.5)
        
        return pixel_values_a, pixel_values_b, torch.tensor(label, dtype=torch.float32)

# --- Model Components ---

class SigLIPFeatureExtractor(nn.Module):
    def __init__(self, model_name=SIGLIP_MODEL_NAME):
        super(SigLIPFeatureExtractor, self).__init__()
        print(f"Loading SigLIP model: {model_name}...")
        self.model = AutoModel.from_pretrained(model_name)
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, pixel_values):
        # SigLIP vision model output
        outputs = self.model.vision_model(pixel_values=pixel_values)
        # Use the pooled output or the embedding of the [CLS] token equivalent
        # SigLIP typically uses the pooled output from the vision model
        # outputs.pooler_output shape: [batch_size, hidden_dim]
        return outputs.pooler_output

class RankNetScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.2):
        super(RankNetScorer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid() # Ensure output is in [0, 1]
        )

    def forward(self, x):
        return self.net(x)

class RankNetLoss(nn.Module):
    def __init__(self, scaling_factor=RANKNET_SCALING_FACTOR):
        super(RankNetLoss, self).__init__()
        self.scaling_factor = scaling_factor
        self.bce_loss = nn.BCELoss()

    def forward(self, score_i, score_j, target_p_ij):
        # score_i, score_j are in [0, 1]
        # score_diff is in [-1, 1]
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Initialize Model
    feature_extractor = SigLIPFeatureExtractor()
    # Get embedding dimension dynamically
    dummy_input = torch.randn(1, 3, 224, 224) # Standard SigLIP input size
    # We need to run a dummy pass or check config. 
    # SigLIP base usually has 768 dim.
    # Let's trust the model config or run a dummy pass on CPU.
    with torch.no_grad():
        dummy_out = feature_extractor.model.vision_model(dummy_input).pooler_output
        input_dim = dummy_out.shape[1]
    
    print(f"Feature dimension: {input_dim}")
    
    scorer = RankNetScorer(input_dim=input_dim)
    model = LegibilityPredictor(feature_extractor, scorer).to(device)
    
    criterion = RankNetLoss(scaling_factor=RANKNET_SCALING_FACTOR)
    optimizer = optim.Adam(model.scorer.parameters(), lr=LEARNING_RATE) # Only train scorer

    # 3. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for pixel_values_a, pixel_values_b, labels in progress_bar:
            pixel_values_a = pixel_values_a.to(device)
            pixel_values_b = pixel_values_b.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            score_a, score_b = model(pixel_values_a, pixel_values_b)
            loss = criterion(score_a, score_b, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_pairs = 0
        total_pairs = 0
        
        with torch.no_grad():
            for pixel_values_a, pixel_values_b, labels in val_loader:
                pixel_values_a = pixel_values_a.to(device)
                pixel_values_b = pixel_values_b.to(device)
                labels = labels.to(device)
                
                score_a, score_b = model(pixel_values_a, pixel_values_b)
                loss = criterion(score_a, score_b, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                # If label > 0.5, expect score_a > score_b
                # If label < 0.5, expect score_b > score_a
                # If label == 0.5, strictly speaking any order is wrong if not equal, 
                # but for accuracy we usually check if sign matches preference.
                # Let's define accuracy as: correct direction prediction.
                
                diff = score_a - score_b
                # Predictions: 1 if a > b, 0 if b > a
                preds = (diff > 0).float()
                
                # Targets for accuracy: 1 if label > 0.5, 0 if label < 0.5
                # Ignore ties for binary accuracy or treat them separately.
                # Here we'll just check standard directional accuracy.
                
                # Create binary targets (1 if A preferred, 0 if B preferred)
                # For ties (0.5), we can't really be "correct" in a binary sense unless we predict exactly 0 diff.
                # Let's exclude ties from accuracy calculation for clarity.
                non_ties = (labels != 0.5)
                if non_ties.sum() > 0:
                    binary_preds = preds[non_ties]
                    binary_targets = (labels[non_ties] > 0.5).float().unsqueeze(1)
                    correct_pairs += (binary_preds == binary_targets).sum().item()
                    total_pairs += non_ties.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_pairs / total_pairs if total_pairs > 0 else 0.0
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc (non-ties): {val_acc:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "legibility_predictor.pth")
            print("Saved best model.")

    # Cleanup temp files
    if os.path.exists("train_temp.jsonl"): os.remove("train_temp.jsonl")
    if os.path.exists("val_temp.jsonl"): os.remove("val_temp.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the JSONL file with ratings.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    args = parser.parse_args()
    
    train(args)
