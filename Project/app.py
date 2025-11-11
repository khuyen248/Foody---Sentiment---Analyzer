from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import pandas as pd
import json
from datetime import datetime, timedelta
import os
import re
from collections import Counter
import numpy as np
from tqdm import tqdm

app = Flask(__name__)
CORS(app)

# Load model và tokenizer - KHỚP VỚI PREDICT.PY
class PhoBERTSentimentClassifier(nn.Module):
    def __init__(self, n_classes=2, dropout=0.3, num_layers_to_unfreeze=4):
        super(PhoBERTSentimentClassifier, self).__init__()
        
        self.phobert = AutoModel.from_pretrained('vinai/phobert-base') 
        
        # Đóng băng các tham số
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

# Global variables
model = None
tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processed_data_cache = None
user_predictions = []  # Lưu trữ các dự đoán của user

def load_model():
    global model, tokenizer
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        
        # Load checkpoint
        checkpoint = torch.load('phobert_foody_best_model.pth', map_location=device)
        
        # Lấy thông tin từ checkpoint
        n_classes = checkpoint.get('n_classes', 2)
        max_length = checkpoint.get('max_length', 128)
        
        # Khởi tạo model với đúng class
        model = PhoBERTSentimentClassifier(n_classes=n_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully! Classes: {n_classes}, Max Length: {max_length}")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text.lower()

def predict_sentiment(text):
    if model is None or tokenizer is None:
        return None, 0.0
    
    try:
        processed_text = preprocess_text(text)
        
        encoding = tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            predictions = torch.nn.functional.softmax(outputs, dim=-1)
            confidence, predicted = torch.max(predictions, 1)
            
            label_map = {0: 'negative', 1: 'positive'}
            sentiment = label_map[predicted.item()]
            confidence_score = confidence.item()
            
            return sentiment, confidence_score
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, 0.0

def load_and_analyze_dataset():
    """Load dataset và phân tích bằng model AI"""
    global processed_data_cache
    
    if processed_data_cache is not None:
        return processed_data_cache
    
    try:
        print("Loading and analyzing dataset...")
        
        # Load tất cả files CSV
        all_data = []
        
        # Load data_test
        if os.path.exists('data_test'):
            for file in os.listdir('data_test'):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join('data_test', file))
                    all_data.append(df)
        
        # Load data_train  
        if os.path.exists('data_train'):
            for file in os.listdir('data_train'):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join('data_train', file))
                    all_data.append(df)
        
        if not all_data:
            print("No CSV files found")
            return None
            
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_data)} reviews")
        
        # Tìm cột text (có thể có tên khác nhau)
        text_columns = ['text', 'comment', 'review', 'content', 'message']
        text_col = None
        for col in text_columns:
            if col in combined_data.columns:
                text_col = col
                break
        
        if text_col is None:
            print("No text column found")
            return None
            
        # Phân tích cảm xúc cho từng review (giới hạn để tránh quá chậm)
        sample_size = min(1000, len(combined_data))  # Phân tích tối đa 1000 reviews
        sample_data = combined_data.sample(n=sample_size, random_state=42).copy()
        
        predictions = []
        confidences = []
        
        print(f"Analyzing {len(sample_data)} reviews with AI model...")
        for idx, row in tqdm(sample_data.iterrows(), total=len(sample_data)):
            text = row[text_col]
            if pd.notna(text) and str(text).strip():
                sentiment, confidence = predict_sentiment(str(text))
                predictions.append(sentiment)
                confidences.append(confidence)
            else:
                predictions.append('neutral')
                confidences.append(0.5)
        
        sample_data['ai_sentiment'] = predictions
        sample_data['ai_confidence'] = confidences
        
        # Tạo dữ liệu tổng hợp
        sentiment_counts = pd.Series(predictions).value_counts()
        total_reviews = len(combined_data)
        
        # Tạo dữ liệu xu hướng theo thời gian từ user predictions thật
        time_series_data = []
        
        if user_predictions:
            # Nhóm predictions theo ngày
            from collections import defaultdict
            daily_counts = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
            
            for pred in user_predictions:
                date = pred['date']
                sentiment = pred['sentiment']
                daily_counts[date][sentiment] += 1
            
            # Tạo dữ liệu cho 6 ngày gần nhất
            sorted_dates = sorted(daily_counts.keys())[-6:] if daily_counts else []
            
            for i, date in enumerate(sorted_dates):
                counts = daily_counts[date]
                total_day = sum(counts.values())
                
                if total_day > 0:
                    time_series_data.append({
                        'month': f"Day {i+1}",
                        'positive': round(counts['positive'] / total_day * 100),
                        'negative': round(counts['negative'] / total_day * 100),
                        'neutral': round(counts['neutral'] / total_day * 100)
                    })
        
        # Nếu không có user predictions, tạo dữ liệu trống
        if not time_series_data:
            for i in range(6):
                time_series_data.append({
                    'month': f"T{i+1}",
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                })
        
        # Trích xuất từ khóa
        all_text = ' '.join(sample_data[text_col].astype(str).tolist()).lower()
        
        # Từ khóa phổ biến trong tiếng Việt cho đánh giá thức ăn
        positive_keywords = [
            'ngon', 'tuyệt', 'tốt', 'nhanh', 'sạch', 'đẹp', 'thích', 'hài lòng', 
            'tuyệt vời', 'xuất sắc', 'chất lượng', 'fresh', 'phục vụ tốt'
        ]
        negative_keywords = [
            'tệ', 'dở', 'chậm', 'bẩn', 'đắt', 'không ngon', 'thất vọng', 'kém',
            'tồi tệ', 'khủng khiếp', 'không tươi', 'phục vụ kém', 'bất cẩn'
        ]
        
        keyword_stats = []
        for keyword in positive_keywords + negative_keywords:
            count = all_text.count(keyword)
            if count > 0:
                sentiment = 'positive' if keyword in positive_keywords else 'negative'
                keyword_stats.append({
                    "keyword": keyword,
                    "count": count,
                    "sentiment": sentiment
                })
        
        keyword_stats.sort(key=lambda x: x['count'], reverse=True)
        
        # Cache kết quả
        processed_data_cache = {
            'total_reviews': total_reviews,
            'sentiment_distribution': sentiment_counts.to_dict(),
            'sample_data': sample_data,
            'time_series': time_series_data,
            'keywords': keyword_stats[:20],
            'text_column': text_col
        }
        
        print("Dataset analysis completed!")
        return processed_data_cache
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        return None

@app.route('/')
def home():
    return jsonify({"message": "Foody Sentiment Analysis API is running!"})

@app.route('/api/predict', methods=['POST'])
def predict():
    global user_predictions
    try:
        data = request.get_json()
        name = data.get("name", '') 
        text = data.get('review', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        sentiment, confidence = predict_sentiment(text)
        
        if sentiment is None:
            return jsonify({"error": "Prediction failed"}), 500
        
        # Lưu prediction vào danh sách user_predictions
        # prediction_record = {
        #     "id": len(user_predictions) + 1,
        #     "text": text,
        #     "sentiment": sentiment,
        #     "confidence": float(confidence),
        #     "timestamp": datetime.now().isoformat(),
        #     "restaurant": "User Input",
        #     "rating": 5 if sentiment == 'positive' else 2,
        #     "date": datetime.now().strftime('%Y-%m-%d')
        # }

        prediction_record = {
    "id": len(user_predictions) + 1,
    "name": data.get("name"), 
    "text": text,
    "sentiment": sentiment,
    "confidence": float(confidence),
    "timestamp": datetime.now().isoformat(),
    "rating": 5 if sentiment == 'positive' else 2,
    "date": datetime.now().strftime('%Y-%m-%d')
}

        user_predictions.append(prediction_record)
        
        return jsonify({
            "text": text,
            "sentiment": sentiment,
            "confidence": float(confidence),
            "timestamp": datetime.now().isoformat(),
            "method": "ai_model"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dataset-stats', methods=['GET'])
def dataset_stats():
    global user_predictions
    try:
        # Kết hợp dữ liệu từ dataset và user predictions
        dataset_data = load_and_analyze_dataset()
        
        # Tính toán từ user predictions
        total_user_predictions = len(user_predictions)
        user_sentiment_counts = {'positive': 0, 'negative': 0}
        
        for pred in user_predictions:
            sentiment = pred['sentiment']
            if sentiment in user_sentiment_counts:
                user_sentiment_counts[sentiment] += 1
        
        # Combine với dataset (nếu có)
        if dataset_data is not None:
            total_reviews = dataset_data['total_reviews'] + total_user_predictions
            combined_sentiment = dataset_data['sentiment_distribution'].copy()
            
            # Cộng thêm user predictions
            for sentiment, count in user_sentiment_counts.items():
                combined_sentiment[sentiment] = combined_sentiment.get(sentiment, 0) + count
                
            sample_reviews = dataset_data['sample_data'].head(5).to_dict('records') if not dataset_data['sample_data'].empty else []
            
            # Thêm user predictions vào sample reviews
            for pred in user_predictions[-5:]:  # 5 predictions gần nhất
                sample_reviews.append({
                    'text': pred['text'],
                    'sentiment': pred['sentiment'],
                    'confidence': pred['confidence'],
                    'name': pred['name'],
                    'rating': pred['rating'],
                    'date': pred['date']
                })
                
        else:
            # Chỉ có user predictions
            total_reviews = total_user_predictions
            combined_sentiment = user_sentiment_counts
            sample_reviews = user_predictions[-10:]  # 10 predictions gần nhất
        
        # Tạo time_series_data từ user predictions
        time_series_data = []
        
        if user_predictions:
            # Nhóm predictions theo ngày
            from collections import defaultdict
            daily_counts = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
            
            for pred in user_predictions:
                date = pred['date']
                sentiment = pred['sentiment']
                daily_counts[date][sentiment] += 1
            
            # Tạo dữ liệu cho các ngày có dữ liệu
            sorted_dates = sorted(daily_counts.keys())
            
            for i, date in enumerate(sorted_dates[-6:]):  # 6 ngày gần nhất
                counts = daily_counts[date]
                total_day = sum(counts.values())
                
                if total_day > 0:
                    time_series_data.append({
                        'month': f"Ngày {i+1}",
                        'positive': round(counts['positive'] / total_day * 100),
                        'negative': round(counts['negative'] / total_day * 100),
                        'neutral': round(counts['neutral'] / total_day * 100)
                    })
        
        # Nếu không có user predictions hoặc không đủ dữ liệu
        if not time_series_data:
            for i in range(6):
                time_series_data.append({
                    'month': f"T{i+1}",
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                })
        
        # Tính phần trăm
        total = sum(combined_sentiment.values()) if combined_sentiment else 0
        
        if total == 0:
            # Không có dữ liệu gì
            positive_percent = 0
            negative_percent = 0
            neutral_percent = 0
            avg_rating = 0
            total_reviews = 0
        else:
            positive_percent = round((combined_sentiment.get('positive', 0) / total * 100), 1)
            negative_percent = round((combined_sentiment.get('negative', 0) / total * 100), 1) 
            neutral_percent = round((combined_sentiment.get('neutral', 0) / total * 100), 1)
            
            # Tính điểm trung bình từ user predictions
            if user_predictions:
                avg_rating = sum(pred['rating'] for pred in user_predictions) / len(user_predictions)
            else:
                avg_rating = 0
        
        return jsonify({
            "total_reviews": total_reviews,
            "positive_percent": positive_percent,
            "negative_percent": negative_percent, 
            "neutral_percent": neutral_percent,
            "avg_rating": round(avg_rating, 1),
            "sentiment_distribution": combined_sentiment,
            "sample_reviews": sample_reviews,
            "time_series": time_series_data
        })
    
    except Exception as e:
        print(f"Error in dataset_stats: {e}")
        # Tạo time_series trống cho error case
        empty_time_series = []
        for i in range(6):
            empty_time_series.append({
                'month': f"T{i+1}",
                'positive': 0,
                'negative': 0,
                'neutral': 0
            })
            
        # Trả về dữ liệu chính xác khi có lỗi
        if user_predictions:
            pos_count = sum(1 for p in user_predictions if p['sentiment'] == 'positive')
            neg_count = sum(1 for p in user_predictions if p['sentiment'] == 'negative')
            total_preds = len(user_predictions)
            
            return jsonify({
                "total_reviews": total_preds,
                "positive_percent": round(pos_count / total_preds * 100, 1),
                "negative_percent": round(neg_count / total_preds * 100, 1),
                "neutral_percent": 0,
                "avg_rating": round(sum(p['rating'] for p in user_predictions) / total_preds, 1),
                "sentiment_distribution": {"positive": pos_count, "negative": neg_count},
                "sample_reviews": user_predictions,
                "time_series": empty_time_series
            })
        else:
            return jsonify({
                "total_reviews": 0,
                "positive_percent": 0,
                "negative_percent": 0,
                "neutral_percent": 0,
                "avg_rating": 0,
                "sentiment_distribution": {"positive": 0, "negative": 0},
                "sample_reviews": [],
                "time_series": empty_time_series
            })

@app.route('/api/keywords', methods=['GET'])
def get_keywords():
    global user_predictions
    try:
        # Kết hợp từ khóa từ dataset và user predictions
        dataset_data = load_and_analyze_dataset()
        
        all_keywords = []
        
        # Trích xuất từ khóa từ user predictions
        if user_predictions:
            user_text = ' '.join([pred['text'] for pred in user_predictions]).lower()
            
            # Từ khóa phổ biến trong tiếng Việt cho đánh giá thức ăn
            positive_keywords = [
                'ngon', 'tuyệt', 'tốt', 'nhanh', 'sạch', 'đẹp', 'thích', 'hài lòng', 
                'tuyệt vời', 'xuất sắc', 'chất lượng', 'fresh', 'phục vụ tốt', 'tươi'
            ]
            negative_keywords = [
                'tệ', 'dở', 'chậm', 'bẩn', 'đắt', 'không ngon', 'thất vọng', 'kém',
                'tồi tệ', 'khủng khiếp', 'không tươi', 'phục vụ kém', 'bất cẩn', 'dở tệ'
            ]
            
            # Đếm từ khóa trong user predictions
            for keyword in positive_keywords + negative_keywords:
                count = user_text.count(keyword)
                if count > 0:
                    sentiment = 'positive' if keyword in positive_keywords else 'negative'
                    all_keywords.append({
                        "keyword": keyword,
                        "count": count,
                        "sentiment": sentiment,
                        "source": "user_input"
                    })
        
        # Thêm từ khóa từ dataset (nếu có)
        if dataset_data and dataset_data.get('keywords'):
            for keyword_data in dataset_data['keywords']:
                # Kiểm tra xem từ khóa đã có từ user predictions chưa
                existing = next((k for k in all_keywords if k['keyword'] == keyword_data['keyword']), None)
                if existing:
                    # Cộng dồn count nếu từ khóa đã tồn tại
                    existing['count'] += keyword_data['count']
                    existing['source'] = "both"
                else:
                    # Thêm từ khóa mới từ dataset
                    keyword_copy = keyword_data.copy()
                    keyword_copy['source'] = "dataset"
                    all_keywords.append(keyword_copy)
        
        # Sắp xếp theo count giảm dần
        all_keywords.sort(key=lambda x: x['count'], reverse=True)
        
        return jsonify({
            "keywords": all_keywords[:20],  # Top 20 keywords
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error in get_keywords: {e}")
        # Fallback: chỉ trả về từ user predictions
        if user_predictions:
            user_text = ' '.join([pred['text'] for pred in user_predictions]).lower()
            basic_keywords = []
            
            basic_positive = ['ngon', 'tốt', 'thích', 'tuyệt']
            basic_negative = ['tệ', 'dở', 'chậm', 'không ngon']
            
            for keyword in basic_positive + basic_negative:
                count = user_text.count(keyword)
                if count > 0:
                    sentiment = 'positive' if keyword in basic_positive else 'negative'
                    basic_keywords.append({
                        "keyword": keyword,
                        "count": count,
                        "sentiment": sentiment
                    })
            
            return jsonify({
                "keywords": basic_keywords,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "keywords": [],
                "timestamp": datetime.now().isoformat()
            })

@app.route('/api/reviews', methods=['GET'])
def get_reviews():
    global user_predictions
    try:
        # Kết hợp dataset và user predictions
        dataset_data = load_and_analyze_dataset()
        
        all_reviews = []
        
        # Thêm user predictions trước (mới nhất)
        for pred in reversed(user_predictions):  # Reverse để mới nhất lên đầu
            all_reviews.append({
                'id': pred['id'],
                'text': pred['text'],
                'sentiment': pred['sentiment'],
                'confidence': pred['confidence'], 
                'restaurant': pred['restaurant'],
                'rating': pred['rating'],
                'date': pred['date'],
                'source': 'User Input'
            })
        
        # Thêm dataset reviews (nếu có)
        if dataset_data is not None:
            for idx, row in dataset_data['sample_data'].head(20).iterrows():
                all_reviews.append({
                    'id': len(user_predictions) + idx + 1,
                    'text': row[dataset_data['text_column']],
                    'sentiment': row['ai_sentiment'],
                    'confidence': row['ai_confidence'],
                    'restaurant': row.get('restaurant', f'Restaurant {idx+1}'),
                    'rating': row.get('rating', np.random.randint(1, 6)),
                    'date': row.get('date', datetime.now().strftime('%Y-%m-%d')),
                    'source': 'Dataset'
                })
        
        return jsonify({"reviews": all_reviews})
    
    except Exception as e:
        # Fallback: chỉ trả về user predictions
        return jsonify({"reviews": user_predictions})

@app.route('/api/analyze-all', methods=['POST'])
def analyze_all():
    """Force reload và phân tích lại toàn bộ dataset"""
    global processed_data_cache
    processed_data_cache = None  # Clear cache
    
    data = load_and_analyze_dataset()
    
    if data is None:
        return jsonify({"error": "Failed to analyze dataset"}), 500
    
    return jsonify({
        "message": "Dataset analyzed successfully",
        "total_reviews": data['total_reviews'],
        "processed_reviews": len(data['sample_data']),
        "sentiment_distribution": data['sentiment_distribution']
    })

if __name__ == '__main__':
    print("Loading model...")
    model_loaded = load_model()
    
    if model_loaded:
        print("Model loaded successfully!")
    else:
        print("Model loading failed.")
    
    print("Starting Flask server...")
    app.run(debug=True, port=5000)