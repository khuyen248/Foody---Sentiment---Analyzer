# main.py (hoặc train_and_evaluate.py)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import glob
import random
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# --- Cấu hình Ban đầu ---
warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# -------------------------

## ----------------------------------------------------
## 1. CUSTOM PYTORCH DATASET
## ----------------------------------------------------

class FoodySentimentDataset(Dataset):
    """Custom Dataset cho đánh giá Foody"""
    
    def __init__(self, texts, labels, tokenizer, max_length): # max_length được truyền vào
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

## ----------------------------------------------------
## 2. PHOBERT CLASSIFIER MODEL
## ----------------------------------------------------

class PhoBERTSentimentClassifier(nn.Module):
    """PhoBERT model cho phân tích cảm xúc với Layer-wise Unfreezing"""
    
    def __init__(self, n_classes=2, dropout=0.3, num_layers_to_unfreeze=4):
        super(PhoBERTSentimentClassifier, self).__init__()
        
        self.phobert = AutoModel.from_pretrained('vinai/phobert-base')
        
        # 1. Freeze TẤT CẢ các tham số
        for param in self.phobert.parameters():
            param.requires_grad = False
            
        # 2. UNFREEZE N layer cuối và Embedding layer
        num_layers = len(self.phobert.encoder.layer) 
        for i in range(1, num_layers_to_unfreeze + 1):
            target_layer = self.phobert.encoder.layer[num_layers - i]
            for param in target_layer.parameters():
                param.requires_grad = True
                
        for param in self.phobert.embeddings.parameters():
            param.requires_grad = True

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.dropout(pooled_output)
        return self.classifier(output)

## ----------------------------------------------------
## 3. MAIN ANALYZER CLASS
## ----------------------------------------------------

class FoodySentimentAnalyzer:
    """Main class để train, đánh giá và predict sentiment"""
    
    def __init__(self, model_name='vinai/phobert-base', max_length=128): # Mặc định là 128
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_map = {0: 'Tiêu cực', 1: 'Tích cực'}
        self.model = None
        
        print(f"Device: {self.device} | Max Length: {self.max_length}")
    
    def load_data_from_folders(self, data_dir):
        # ... (Hàm này giữ nguyên như code trước)
        def read_files_from_dir(directory, label):
            if not os.path.exists(directory): 
                print(f"Cảnh báo: Thư mục không tồn tại: {directory}")
                return [], []
                
            files = glob.glob(os.path.join(directory, '*.txt'))
            dir_texts, dir_labels = [], []
            
            for file_path in tqdm(files, desc=f"Loading {os.path.basename(directory)}"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            dir_texts.append(content)
                            dir_labels.append(label)
                except Exception: 
                    continue
            return dir_texts, dir_labels
        
        # Load training data
        train_texts, train_labels = [], []
        train_pos_dir = os.path.join(data_dir, 'train', 'pos')
        train_neg_dir = os.path.join(data_dir, 'train', 'neg')
        pos_texts, pos_labels = read_files_from_dir(train_pos_dir, 1) 
        neg_texts, neg_labels = read_files_from_dir(train_neg_dir, 0) 
        train_texts.extend(pos_texts); train_labels.extend(pos_labels)
        train_texts.extend(neg_texts); train_labels.extend(neg_labels)
        
        # Load test data
        test_texts, test_labels = [], []
        test_pos_dir = os.path.join(data_dir, 'test', 'pos')
        test_neg_dir = os.path.join(data_dir, 'test', 'neg')
        pos_texts, pos_labels = read_files_from_dir(test_pos_dir, 1)
        neg_texts, neg_labels = read_files_from_dir(test_neg_dir, 0)
        test_texts.extend(pos_texts); test_labels.extend(pos_labels)
        test_texts.extend(neg_texts); test_labels.extend(neg_labels)
        
        if len(train_texts) == 0:
            raise ValueError("Không có dữ liệu training được load!")
        
        print(f"\n[Dữ liệu] Train: {len(train_texts)} samples | Test: {len(test_texts)} samples")
        return {
            'train_texts': train_texts, 'train_labels': train_labels,
            'test_texts': test_texts, 'test_labels': test_labels
        }
    
    def prepare_data_loaders(self, texts, labels, test_size=0.1, batch_size=4, is_test_set=False):
        """Chuẩn bị data loaders, chia train/validation nếu cần"""
        
        if not is_test_set and test_size > 0:
            # Chia train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                texts, labels, test_size=test_size, random_state=SEED, stratify=labels
            )
            # Truyền self.max_length đã được cập nhật
            train_dataset = FoodySentimentDataset(X_train, y_train, self.tokenizer, self.max_length)
            val_dataset = FoodySentimentDataset(X_val, y_val, self.tokenizer, self.max_length)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            return train_loader, val_loader, len(X_train), len(X_val)
        
        # Dùng cho Test set hoặc Full Training set
        dataset = FoodySentimentDataset(texts, labels, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader, None, len(texts), 0
    
    def train_model(self, train_loader, val_loader=None, epochs=5, learning_rate=2e-5, save_path='phobert_foody_best_model.pth'):
        """Training model với logic lưu model tốt nhất"""
        
        model = PhoBERTSentimentClassifier(n_classes=2).to(self.device)
        
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=learning_rate
        )
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('=' * 50)
            
            # Training phase
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Training E{epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_acc = self._evaluate_accuracy(model, val_loader)
                val_accuracies.append(val_acc)
                print(f'Training Loss: {avg_train_loss:.4f} | Validation Accuracy: {val_acc:.4f}')
                
                # LƯU MODEL TỐT NHẤT
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Lưu kèm max_length đã dùng
                    self.save_model(save_path, model, self.max_length) 
                    print(f"--> Model đã được lưu vào '{save_path}' (Best Val Acc: {best_val_acc:.4f})")
            else:
                print(f'Training Loss: {avg_train_loss:.4f}')
        
        self.model = model 
        
        return train_losses, val_accuracies
    
    # ... (các hàm _evaluate_accuracy, evaluate_model_detailed, predict_sentiment, plot_confusion_matrix giữ nguyên)
    def _evaluate_accuracy(self, model, data_loader):
        """Hàm nội bộ đánh giá accuracy"""
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        return correct_predictions / total_predictions
        
    def evaluate_model_detailed(self, data_loader):
        """Đánh giá chi tiết model (Report & Confusion Matrix)"""
        if self.model is None:
            raise Exception("Model chưa được huấn luyện hoặc load!")
            
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating Test Set"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        target_names = [self.label_map[0], self.label_map[1]]
        report = classification_report(
            all_labels, all_predictions, 
            target_names=target_names, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {'accuracy': accuracy, 'classification_report': report, 'confusion_matrix': cm}
    
    def predict_sentiment(self, text):
        """Dự đoán cảm xúc cho một văn bản (Đảm bảo model đã được load vào self.model)"""
        if self.model is None:
            raise Exception("Model chưa được load. Vui lòng chạy load_model() trước.")
            
        self.model.eval()
        
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return {
            'sentiment': self.label_map[predicted.item()],
            'confidence': confidence.item()
        }
    
    def save_model(self, path, model, max_length):
        """Lưu model (chỉ state_dict) kèm theo max_length"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer_name': self.model_name,
            'max_length': max_length, # LƯU MAX_LENGTH ĐÃ DÙNG KHI TRAIN
            'n_classes': 2
        }, path)
    
    def load_model(self, path):
        """Load model đã lưu bằng state_dict"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file model: {path}.")

        checkpoint = torch.load(path, map_location=self.device)
        
        self.model = PhoBERTSentimentClassifier(n_classes=2).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval() 
        
        self.model_name = checkpoint.get('tokenizer_name', 'vinai/phobert-base')
        self.max_length = checkpoint.get('max_length', 128) # Cập nhật max_length sau khi load
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print(f"Model loaded from {path}. Max Length: {self.max_length}")

    def plot_confusion_matrix(self, cm, title='Test Set Confusion Matrix'):
        """Vẽ confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Tiêu cực', 'Tích cực'],
                    yticklabels=['Tiêu cực', 'Tích cực'])
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

## ----------------------------------------------------
## 4. EXECUTION (MAIN FUNCTION)
## ----------------------------------------------------

def main_train_and_evaluate():
    """Chạy quy trình training và đánh giá hoàn chỉnh"""
    
    # --- Cấu hình Hyperparameter và Đường dẫn ---
    DATA_DIR = r'data_train\data_train'  # THAY THẾ bằng đường dẫn thư mục gốc của bạn
    MODEL_SAVE_PATH = 'phobert_foody_best_model.pth'
    
    EPOCHS = 5
    # TỐI ƯU VRAM 4GB
    BATCH_SIZE = 4      # GIẢM TỐI ĐA BATCH SIZE
    MAX_LENGTH = 128    # GIẢM TỐI ĐA MAX LENGTH
    LEARNING_RATE = 2e-5
    # ----------------------------------------------
    
    analyzer = FoodySentimentAnalyzer(max_length=MAX_LENGTH)
    
    try:
        # 1. Load Data
        data = analyzer.load_data_from_folders(DATA_DIR)
        
        # 2. Chuẩn bị Data Loaders (Sử dụng BATCH_SIZE đã tối ưu)
        train_loader, val_loader, train_size, val_size = analyzer.prepare_data_loaders(
            data['train_texts'], data['train_labels'], batch_size=BATCH_SIZE, test_size=0.1
        )
        test_loader, _, test_size, _ = analyzer.prepare_data_loaders(
            data['test_texts'], data['test_labels'], batch_size=BATCH_SIZE, is_test_set=True
        )
        
        print(f"\nTraining set: {train_size} | Validation set: {val_size} | Test set: {test_size}")
        
        # 3. Train Model
        print("\n" + "="*50)
        print(f"BẮT ĐẦU TRAINING VỚI BATCH={BATCH_SIZE}, MAX_LENGTH={MAX_LENGTH}")
        print("="*50)
        
        analyzer.train_model(
            train_loader, val_loader, epochs=EPOCHS, 
            learning_rate=LEARNING_RATE, save_path=MODEL_SAVE_PATH
        )
        
        # 4. Load lại Model Tốt nhất để đánh giá
        print("\n" + "="*50)
        print("TẢI LẠI MODEL TỐT NHẤT VÀ ĐÁNH GIÁ CUỐI CÙNG")
        print("="*50)
        
        if os.path.exists(MODEL_SAVE_PATH):
             analyzer.load_model(MODEL_SAVE_PATH)
        else:
             raise FileNotFoundError(f"Không tìm thấy model tốt nhất đã lưu tại {MODEL_SAVE_PATH}.")
        
        # 5. Đánh giá trên Test Set
        test_results = analyzer.evaluate_model_detailed(test_loader)
        
        print(f"\nTest Accuracy Cuối Cùng: {test_results['accuracy']:.4f}")
        
        # In Classification Report
        report = test_results['classification_report']
        print(f"\nClassification Report:")
        print(f"  Tiêu cực (Negative): F1={report['Tiêu cực']['f1-score']:.4f}")
        print(f"  Tích cực (Positive): F1={report['Tích cực']['f1-score']:.4f}")
        
        # Vẽ Confusion Matrix
        analyzer.plot_confusion_matrix(test_results['confusion_matrix'])
        
        # 6. Test Predictions
        # ... (Phần test dự đoán mẫu)

    except FileNotFoundError:
        print(f"\nLỖI: Không tìm thấy thư mục dữ liệu tại đường dẫn: {DATA_DIR}")
    except Exception as e:
        print(f"\nLỖI KHÔNG MONG MUỐN: {e}")

if __name__ == "__main__":
    main_train_and_evaluate()