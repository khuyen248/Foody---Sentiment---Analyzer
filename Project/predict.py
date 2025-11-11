# predict.py - Đã tối ưu VRAM (sử dụng FP16)

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm
import warnings

# --- Cấu hình Ban đầu ---
warnings.filterwarnings('ignore')
MODEL_LOAD_PATH = 'phobert_foody_best_model.pth' 
# -------------------------

## ----------------------------------------------------
## 1. PHOBERT CLASSIFIER MODEL (Kiến trúc)
## ----------------------------------------------------
# Lớp này PHẢI giống hệt kiến trúc đã dùng khi train.

class PhoBERTSentimentClassifier(nn.Module):
    
    def __init__(self, n_classes=2, dropout=0.3, num_layers_to_unfreeze=4):
        super(PhoBERTSentimentClassifier, self).__init__()
        
        self.phobert = AutoModel.from_pretrained('vinai/phobert-base') 
        
        # Đóng băng các tham số (chỉ giữ kiến trúc, không train)
        for param in self.phobert.parameters():
            param.requires_grad = False
            
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
## 2. PREDICTION ANALYZER CLASS (Tối ưu FP16)
## ----------------------------------------------------

class PredictionAnalyzer:
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.max_length = 128 # Đặt mặc định an toàn
        self.label_map = {0: 'Tiêu cực', 1: 'Tích cực'}
        
        print(f"Prediction Device: {self.device}")

    def load_model(self, path):
        """Tải model đã được lưu (state_dict) với Half-Precision (FP16) trên GPU"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file model: {path}. Vui lòng chạy main.py trước.")
            
        checkpoint = torch.load(path, map_location=self.device)
        
        # Quyết định dùng FP16 hay không
        use_half_precision = self.device.type == 'cuda' and torch.cuda.is_available()

        # 1. Khởi tạo model và tải weights
        n_classes = checkpoint.get('n_classes', 2) 
        self.max_length = checkpoint.get('max_length', 128) # Lấy max_length đã train (128)
        
        self.model = PhoBERTSentimentClassifier(n_classes=n_classes).to(self.device)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        
        # CHUYỂN SANG HALF-PRECISION (FP16) NẾU DÙNG GPU
        if use_half_precision:
            self.model.half()
            print("--> Đã kích hoạt chế độ Half-Precision (FP16) để tiết kiệm VRAM.")
            
        self.model.eval()
        
        # 2. Khởi tạo tokenizer và tham số
        model_name = checkpoint.get('tokenizer_name', 'vinai/phobert-base')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Model '{path}' đã được tải thành công. Max Length: {self.max_length}. Sẵn sàng dự đoán.")

    def predict_sentiment(self, text):
        """Dự đoán cảm xúc cho một văn bản duy nhất"""
        if self.model is None or self.tokenizer is None:
            raise Exception("Model chưa được load. Vui lòng chạy load_model() trước.")
            
        # Tokenize văn bản
        encoding = self.tokenizer(
            text, 
            add_special_tokens=True, 
            max_length=self.max_length,
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Đảm bảo input đúng dtype nếu model là FP16
        if next(self.model.parameters()).dtype == torch.float16:
            # Token IDs và Attention Mask phải luôn là LongTensor
            pass 
        else:
            # Nếu model không phải FP16, input vẫn giữ nguyên
            pass
            
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return {
            'sentiment': self.label_map[predicted.item()],
            'confidence': confidence.item()
        }

    def batch_predict(self, texts):
        """Dự đoán cho một danh sách văn bản"""
        results = []
        for text in tqdm(texts, desc="Predicting Batch"):
            results.append(self.predict_sentiment(text))
        return results

## ----------------------------------------------------
## 3. EXECUTION
## ----------------------------------------------------

def main_predict():
    """Chạy quy trình load model và dự đoán mẫu"""
    
    analyzer = PredictionAnalyzer()
    
    try:
        # 1. Load Model đã lưu
        analyzer.load_model(MODEL_LOAD_PATH)
        
        # 2. Các văn bản cần dự đoán
        test_texts = [
            "Món ăn tuyệt vời, không gian đẹp, phục vụ rất nhiệt tình.",
            "Tệ, đồ ăn nguội lạnh và thái độ nhân viên không thể chấp nhận được.",
            "Giá hơi đắt nhưng chất lượng xứng đáng, tôi sẽ cân nhắc quay lại.",
            "Thất vọng hoàn toàn, không có gì đặc biệt cả.",
            "Phở ngon nhưng phục vụ hơi chậm.",
        ]
        
        print("\n" + "="*50)
        print("KẾT QUẢ DỰ ĐOÁN TỪ MODEL ĐÃ LƯU:")
        print("="*50)
        
        for text in test_texts:
            result = analyzer.predict_sentiment(text)
            print(f"[Review] '{text}'")
            print(f"  -> Cảm xúc: {result['sentiment']} | Độ tin cậy: {result['confidence']:.3f}")
            
    except FileNotFoundError as e:
        print(f"\nLỖI KHÔNG TÌM THẤY: {e}")
        print("Vui lòng đảm bảo bạn đã chạy file main.py (hoặc train_and_evaluate.py) để tạo file model.")
    except Exception as e:
        print(f"\nLỖI CHẠY PREDICT: {e}")

if __name__ == "__main__":
    main_predict()