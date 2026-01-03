"""
=============================================================================
HỆ THỐNG TRAIN MÔ HÌNH CHUYỂN TEENCODE VIỆT NAM THÀNH VĂN BẢN CHUẨN
=============================================================================
Tác giả: AI Professor
Mục đích: Text Normalization cho tiếng Việt với teencode
Kiến trúc: Encoder-Decoder based on mBERT + BARTpho
=============================================================================
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_metric
import os
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# BƯỚC 1: PHÂN TÍCH DỮ LIỆU VÀ HIỂU CẤU TRÚC
# =============================================================================
print("="*80)
print("BƯỚC 1: PHÂN TÍCH DỮ LIỆU")
print("="*80)

class DataAnalyzer:
    """
    Lý do: Cần hiểu cấu trúc dữ liệu trước khi xử lý
    Nguyên lý: Exploratory Data Analysis (EDA)
    """
    def __init__(self, data_paths: Dict[str, str]):
        self.data_paths = data_paths
        
    def analyze(self):
        print("\n📊 Đang phân tích dữ liệu...")
        
        for split_name, path in self.data_paths.items():
            df = pd.read_csv(path)
            print(f"\n{split_name.upper()} Dataset:")
            print(f"  - Số lượng mẫu: {len(df)}")
            print(f"  - Các cột: {df.columns.tolist()}")
            print(f"  - Ví dụ:")
            print(df.head(2))
            
            # Phân tích độ dài câu
            if 'input' in df.columns and 'output' in df.columns:
                input_lens = df['input'].str.split().str.len()
                output_lens = df['output'].str.split().str.len()
                print(f"  - Độ dài trung bình input: {input_lens.mean():.1f} từ")
                print(f"  - Độ dài trung bình output: {output_lens.mean():.1f} từ")

# =============================================================================
# BƯỚC 2: XỬ LÝ DỮ LIỆU VÀ CHUẨN HÓA
# =============================================================================
print("\n" + "="*80)
print("BƯỚC 2: XỬ LÝ VÀ CHUẨN HÓA DỮ LIỆU")
print("="*80)

class VietnameseTextProcessor:
    """
    Lý do: Dữ liệu thô cần được làm sạch và chuẩn hóa
    Chức năng:
      1. Xử lý khoảng trắng thừa
      2. Chuẩn hóa ký tự đặc biệt
      3. Giữ nguyên emoji và icon (vì có ý nghĩa trong chat)
    """
    
    @staticmethod
    def normalize_spaces(text: str) -> str:
        """Chuẩn hóa khoảng trắng"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Làm sạch văn bản nhưng giữ nguyên cấu trúc chat"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Giữ emoji và ký tự đặc biệt (quan trọng cho ngữ cảnh chat)
        text = VietnameseTextProcessor.normalize_spaces(text)
        return text.lower()  # Chuyển về lowercase để dễ học

# =============================================================================
# BƯỚC 3: TẠO DATASET CLASS
# =============================================================================
print("\n" + "="*80)
print("BƯỚC 3: XÂY DỰNG PYTORCH DATASET")
print("="*80)

class TeencodeDataset(Dataset):
    """
    Lý do: PyTorch yêu cầu Dataset class để load dữ liệu hiệu quả
    Nguyên lý: 
      - __len__: Trả về số lượng mẫu
      - __getitem__: Trả về 1 mẫu theo index
    Chức năng:
      - Tokenize input (teencode)
      - Tokenize output (văn bản chuẩn) với labels
      - Padding và truncation
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processor = VietnameseTextProcessor()
        
        # Chuẩn hóa dữ liệu
        print(f"📁 Loading data từ: {data_path}")
        self.df['input'] = self.df['input'].apply(self.processor.clean_text)
        self.df['output'] = self.df['output'].apply(self.processor.clean_text)
        
        print(f"✅ Đã load {len(self.df)} mẫu")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_text = row['input']
        target_text = row['output']
        
        # Tokenize input
        # Lý do: Mô hình chỉ hiểu số, không hiểu chữ
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        # Lý do: Decoder cần labels để tính loss
        labels = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Quan trọng: Thay padding token bằng -100 để không tính loss
        # Nguyên lý: CrossEntropyLoss bỏ qua label=-100
        labels_ids = labels['input_ids'].squeeze()
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': model_inputs['input_ids'].squeeze(),
            'attention_mask': model_inputs['attention_mask'].squeeze(),
            'labels': labels_ids
        }

# =============================================================================
# BƯỚC 4: CHỌN VÀ CẤU HÌNH MÔ HÌNH
# =============================================================================
print("\n" + "="*80)
print("BƯỚC 4: CHỌN VÀ CẤU HÌNH MÔ HÌNH")
print("="*80)

class ModelSelector:
    """
    Lý do: Cần chọn mô hình phù hợp với tiếng Việt
    
    So sánh các lựa chọn:
    
    1. BARTpho (vinai/bartpho-syllable):
       ✅ Pre-trained trên tiếng Việt
       ✅ Hiểu tokenization theo âm tiết
       ✅ Seq2Seq architecture sẵn
       ❌ Model size lớn hơn
       
    2. mBART (facebook/mbart-large-50):
       ✅ Multilingual, có tiếng Việt
       ✅ Mạnh về translation
       ❌ Cần fine-tune nhiều hơn
       
    3. mT5 (google/mt5-base):
       ✅ Text-to-Text framework
       ✅ Multilingual
       ❌ Không chuyên về Việt
    
    QUYẾT ĐỊNH: Dùng BARTpho vì:
    - Pre-trained trên corpus tiếng Việt lớn
    - Tokenizer âm tiết phù hợp với Việt
    - Architecture Seq2Seq sẵn có
    """
    
    @staticmethod
    def get_model_and_tokenizer(model_name: str = "vinai/bartpho-syllable"):
        print(f"\n🤖 Đang load model: {model_name}")
        print("\nNguyên lý hoạt động của BART:")
        print("  - Encoder: Biến input thành hidden representations")
        print("  - Decoder: Sinh output từ representations + attention")
        print("  - Cross-attention: Decoder nhìn vào encoder outputs")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        print(f"✅ Model đã được load")
        print(f"  - Số parameters: {model.num_parameters():,}")
        print(f"  - Vocab size: {tokenizer.vocab_size:,}")
        
        return model, tokenizer

# =============================================================================
# BƯỚC 5: THIẾT LẬP METRICS ĐỂ ĐÁNH GIÁ
# =============================================================================
print("\n" + "="*80)
print("BƯỚC 5: THIẾT LẬP METRICS")
print("="*80)

class MetricsComputer:
    """
    Lý do: Cần đo lường chất lượng của model
    
    Metrics sử dụng:
    
    1. BLEU (Bilingual Evaluation Understudy):
       - Đo độ tương đồng n-gram giữa prediction và reference
       - BLEU-1: unigram (từ đơn)
       - BLEU-2: bigram (cặp từ)
       - Công thức: BLEU = BP × exp(Σ wn log pn)
         + BP: Brevity Penalty (phạt câu quá ngắn)
         + pn: Precision của n-gram
       
    2. Character Error Rate (CER):
       - Đo edit distance ở level ký tự
       - CER = (S + D + I) / N
         + S: Substitutions
         + D: Deletions  
         + I: Insertions
         + N: Tổng số ký tự trong reference
       - Quan trọng cho tiếng Việt vì dấu thanh
    
    3. Word Error Rate (WER):
       - Tương tự CER nhưng ở level từ
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bleu = load_metric("sacrebleu")
        
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        # Decode predictions
        # Lý do: Model output là token ids, cần chuyển về text
        decoded_preds = self.tokenizer.batch_decode(
            predictions, 
            skip_special_tokens=True
        )
        
        # Decode labels (thay -100 về pad token trước)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, 
            skip_special_tokens=True
        )
        
        # Tính BLEU
        # Lý do: BLEU là metric phổ biến cho machine translation
        result = self.bleu.compute(
            predictions=decoded_preds, 
            references=[[label] for label in decoded_labels]
        )
        
        # Tính accuracy đơn giản (exact match)
        exact_match = sum([
            pred.strip() == label.strip() 
            for pred, label in zip(decoded_preds, decoded_labels)
        ]) / len(decoded_preds)
        
        return {
            'bleu': result['score'],
            'exact_match': exact_match * 100
        }

# =============================================================================
# BƯỚC 6: CẤU HÌNH TRAINING
# =============================================================================
print("\n" + "="*80)
print("BƯỚC 6: CẤU HÌNH TRAINING")
print("="*80)

class TrainingConfigurator:
    """
    Lý do: Cần config hyperparameters phù hợp
    
    Giải thích các tham số quan trọng:
    
    1. Learning Rate (2e-5):
       - Quá cao: Model không hội tụ (oscillate)
       - Quá thấp: Học chậm, có thể bị stuck
       - 2e-5: Giá trị tốt cho fine-tuning BERT-based models
    
    2. Batch Size (8-16):
       - Lớn: Gradient stable hơn, nhưng tốn RAM
       - Nhỏ: Tốn ít RAM, nhưng noisy gradient
       - Gradient accumulation: Trick để tăng effective batch size
    
    3. Number of Epochs (10-15):
       - Quá ít: Underfitting
       - Quá nhiều: Overfitting
       - Early stopping: Dừng khi validation không cải thiện
    
    4. Weight Decay (0.01):
       - L2 regularization để tránh overfitting
       - Công thức: Loss = Original_Loss + λ × Σ(weights²)
    
    5. Warmup Steps:
       - Tăng learning rate dần từ 0 lên max
       - Giúp training stable hơn ở đầu
    """
    
    @staticmethod
    def get_training_args(output_dir: str = "./teencode_model"):
        print("\n⚙️ Cấu hình Training Arguments:")
        
        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            
            # Training schedule
            num_train_epochs=15,  # Số epoch
            per_device_train_batch_size=8,  # Batch size cho training
            per_device_eval_batch_size=8,   # Batch size cho evaluation
            
            # Optimizer settings
            learning_rate=2e-5,  # Learning rate
            weight_decay=0.01,   # L2 regularization
            warmup_steps=500,    # Warmup learning rate
            
            # Evaluation và logging
            eval_strategy="steps",  # Evaluate mỗi N steps
            eval_steps=500,         # Evaluate mỗi 500 steps
            save_steps=500,         # Save checkpoint mỗi 500 steps
            logging_steps=100,      # Log mỗi 100 steps
            
            # Early stopping và best model
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            save_total_limit=3,  # Chỉ giữ 3 checkpoints tốt nhất
            
            # Generation settings cho evaluation
            predict_with_generate=True,
            generation_max_length=128,
            generation_num_beams=4,  # Beam search với 4 beams
            
            # Mixed precision training (tăng tốc và giảm RAM)
            fp16=torch.cuda.is_available(),
            
            # Gradient accumulation (để tăng effective batch size)
            gradient_accumulation_steps=2,
            
            # Report to
            report_to="none",  # Có thể dùng "wandb" nếu muốn track
        )
        
        print(f"  ✓ Output directory: {output_dir}")
        print(f"  ✓ Epochs: {args.num_train_epochs}")
        print(f"  ✓ Learning rate: {args.learning_rate}")
        print(f"  ✓ Batch size: {args.per_device_train_batch_size}")
        print(f"  ✓ Mixed precision: {args.fp16}")
        
        return args

# =============================================================================
# BƯỚC 7: XÂY DỰNG TRAINING PIPELINE
# =============================================================================
print("\n" + "="*80)
print("BƯỚC 7: XÂY DỰNG TRAINING PIPELINE")
print("="*80)

class TeencodeTrainer:
    """
    Lý do: Tổ chức toàn bộ quá trình training
    
    Pipeline hoạt động:
    1. Load model và tokenizer
    2. Load và preprocess data
    3. Setup trainer
    4. Train model
    5. Evaluate
    6. Save model
    """
    
    def __init__(self, data_paths: Dict[str, str]):
        self.data_paths = data_paths
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️  Device: {self.device}")
        
    def train(self):
        """Main training function"""
        
        # Step 1: Analyze data
        print("\n" + "="*80)
        print("STEP 1/7: PHÂN TÍCH DỮ LIỆU")
        print("="*80)
        analyzer = DataAnalyzer(self.data_paths)
        analyzer.analyze()
        
        # Step 2: Load model
        print("\n" + "="*80)
        print("STEP 2/7: LOAD MODEL VÀ TOKENIZER")
        print("="*80)
        model, tokenizer = ModelSelector.get_model_and_tokenizer()
        model = model.to(self.device)
        
        # Step 3: Create datasets
        print("\n" + "="*80)
        print("STEP 3/7: TẠO DATASETS")
        print("="*80)
        train_dataset = TeencodeDataset(
            self.data_paths['train'], 
            tokenizer, 
            max_length=128
        )
        eval_dataset = TeencodeDataset(
            self.data_paths['dev'], 
            tokenizer, 
            max_length=128
        )
        
        # Step 4: Setup data collator
        print("\n" + "="*80)
        print("STEP 4/7: SETUP DATA COLLATOR")
        print("="*80)
        print("Data Collator:")
        print("  - Động: Padding chỉ đến max length của batch (tiết kiệm RAM)")
        print("  - Label smoothing: Có thể thêm để tránh overconfident")
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )
        
        # Step 5: Setup metrics
        print("\n" + "="*80)
        print("STEP 5/7: SETUP METRICS")
        print("="*80)
        metrics_computer = MetricsComputer(tokenizer)
        
        # Step 6: Setup training arguments
        print("\n" + "="*80)
        print("STEP 6/7: CẤU HÌNH TRAINING")
        print("="*80)
        training_args = TrainingConfigurator.get_training_args()
        
        # Step 7: Create trainer và train
        print("\n" + "="*80)
        print("STEP 7/7: BẮT ĐẦU TRAINING")
        print("="*80)
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=metrics_computer.compute_metrics,
        )
        
        print("\n🚀 Bắt đầu training...")
        print("\nQuá trình training:")
        print("  1. Forward pass: Input → Encoder → Decoder → Logits")
        print("  2. Loss computation: CrossEntropyLoss(logits, labels)")
        print("  3. Backward pass: Tính gradients")
        print("  4. Optimizer step: Update weights")
        print("  5. Repeat cho mỗi batch")
        
        # Train
        train_result = trainer.train()
        
        # Save final model
        print("\n" + "="*80)
        print("💾 SAVING MODEL")
        print("="*80)
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        print(f"✅ Model đã được save tại: {training_args.output_dir}")
        
        # Evaluate on test set
        print("\n" + "="*80)
        print("📊 EVALUATION ON TEST SET")
        print("="*80)
        test_dataset = TeencodeDataset(
            self.data_paths['test'], 
            tokenizer, 
            max_length=128
        )
        test_results = trainer.evaluate(test_dataset)
        
        print("\n📈 Test Results:")
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}")
        
        return trainer, test_results

# =============================================================================
# BƯỚC 8: INFERENCE VÀ DEMO
# =============================================================================

class TeencodeNormalizer:
    """
    Lý do: Sử dụng model đã train để normalize teencode
    
    Beam Search Decoding:
      - Không chọn từ có xác suất cao nhất mỗi bước (greedy)
      - Giữ k hypotheses tốt nhất (k=num_beams)
      - Chọn sequence có probability cao nhất tổng thể
      - Ví dụ với beam=3:
        Step 1: "tôi" | "mình" | "em"
        Step 2: "tôi đang" | "mình đang" | "tôi đi"
        ...
    """
    
    def __init__(self, model_path: str):
        print(f"\n🔄 Loading model từ {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✅ Model ready!")
        
    def normalize(self, text: str, num_beams: int = 4) -> str:
        """
        Normalize teencode text
        
        Args:
            text: Input teencode
            num_beams: Số beams cho beam search (càng lớn càng tốt nhưng chậm)
        """
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            max_length=128, 
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,  # Tránh lặp n-gram
                length_penalty=1.0,       # Penalty cho độ dài
            )
        
        # Decode
        normalized = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return normalized

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║     🎓 TRAINING TEENCODE NORMALIZATION MODEL FOR VIETNAMESE 🇻🇳        ║
    ║                                                                       ║
    ║     Kiến trúc: BART-based Sequence-to-Sequence                       ║
    ║     Task: Text Normalization (Lexical Normalization)                 ║
    ║     Dataset: ViLexNorm                                               ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Data paths
    data_paths = {
        'train': r'C:\Users\maxez\OneDrive\Documents\DATN-Product-summary\DATN-Product-Summary\datateencode\train.csv',
        'dev': r'C:\Users\maxez\OneDrive\Documents\DATN-Product-summary\DATN-Product-Summary\datateencode\dev.csv',
        'test': r'C:\Users\maxez\OneDrive\Documents\DATN-Product-summary\DATN-Product-Summary\datateencode\test.csv'
    }
    
    # Kiểm tra files tồn tại
    print("\n🔍 Kiểm tra files...")
    for name, path in data_paths.items():
        if os.path.exists(path):
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: File không tồn tại!")
            return
    
    # Initialize trainer
    trainer = TeencodeTrainer(data_paths)
    
    # Train
    model, results = trainer.train()
    
    # Demo inference
    print("\n" + "="*80)
    print("🎯 DEMO INFERENCE")
    print("="*80)
    
    normalizer = TeencodeNormalizer("./teencode_model")
    
    test_cases = [
        "k biet lam sao nua",
        "vs cau ay thi minh cx chiu",
        "ck ay bua qua di",
        "hom wa minh di choi vui vcl"
    ]
    
    print("\nKết quả normalize:")
    for teencode in test_cases:
        normalized = normalizer.normalize(teencode)
        print(f"\n  Input:  {teencode}")
        print(f"  Output: {normalized}")
    
    print("\n" + "="*80)
    print("✅ HOÀN THÀNH!")
    print("="*80)
    print(f"\nModel đã được lưu tại: ./teencode_model")
    print("\nCách sử dụng model:")
    print("""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    tokenizer = AutoTokenizer.from_pretrained('./teencode_model')
    model = AutoModelForSeq2SeqLM.from_pretrained('./teencode_model')
    
    text = "k biet lam sao"
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model.generate(**inputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    """)

if __name__ == "__main__":
    main()