import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm
import json
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = Path(os.environ.get('FYP_DATA_DIR', PROJECT_ROOT / 'dataset')).resolve()
RESULTS_DIR = Path(os.environ.get('FYP_RESULTS_DIR', PROJECT_ROOT / 'results')).resolve()
MODEL_PATH = RESULTS_DIR / 'best_model.pth'
TRAIN_CSV = BASE_DIR / 'train_updated.csv'
TEST_CSV = BASE_DIR / 'test_updated.csv'
LABEL_MAP_PATH = RESULTS_DIR / 'label_mapping.json'

os.makedirs(RESULTS_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def read_csv_with_encoding(file_path):
    encodings = ['utf-8', 'latin1', 'gbk', 'big5', 'gb18030']

    for encoding in encodings:
        try:
            print(f"Trying to read {file_path} with {encoding} encoding...")
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully read with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file with {encoding} encoding: {str(e)}")
            continue

    raise ValueError(f"Unable to read {file_path} with any of the attempted encodings")

def get_training_samples(train_csv_path):

    train_df = read_csv_with_encoding(train_csv_path)
    return dict(Counter(train_df['categoryName']))

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()
        self.resnet = nn.Sequential(
            *list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1]
        )
        self.fc1 = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(0.5)

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc2 = nn.Linear(768, 512)

        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, image, input_ids, attention_mask):
        img_features = self.resnet(image).view(image.size(0), -1)
        img_features = torch.relu(self.fc1(img_features))
        img_features = self.dropout(img_features)

        text_output = self.bert(input_ids, attention_mask=attention_mask)
        text_features = torch.relu(self.fc2(text_output.last_hidden_state[:, 0, :]))
        text_features = self.dropout(text_features)

        img_features = img_features.unsqueeze(1)
        text_features = text_features.unsqueeze(1)
        combined_features = torch.cat((img_features, text_features), dim=1)

        attn_output, _ = self.attention(combined_features, combined_features, combined_features)
        fused_features = attn_output.mean(dim=1)

        output = self.fc3(fused_features)
        return output

class TestDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, tokenizer=None):
        self.data = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['imgUrl'])

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        title = row['title']
        inputs = self.tokenizer(
            title,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(row['label'], dtype=torch.long),
            'category_name': row['categoryName']
        }

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
    return torch.utils.data.dataloader.default_collate(batch)

def calculate_metrics(class_predictions, test_samples, train_samples, min_test_samples=10):

    class_metrics = {}

    for category, stats in class_predictions.items():
        if stats['total'] >= min_test_samples:
            test_acc = stats['correct'] / stats['total']
            test_count = stats['total']
            train_count = train_samples.get(category, 0)

            sample_weight = np.log1p(test_count)
            score = test_acc * sample_weight

            class_metrics[category] = {
                'accuracy': test_acc,
                'test_samples': test_count,
                'train_samples': train_count,
                'correct_predictions': stats['correct'],
                'score': score
            }

    return class_metrics

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label mapping file not found at {LABEL_MAP_PATH}")

    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        saved_label_info = json.load(f)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.asarray(saved_label_info['classes'])
        num_classes = len(label_encoder.classes_)
    print(f"Loaded label mapping with {num_classes} classes")

    print("Loading training set statistics...")
    training_samples = get_training_samples(TRAIN_CSV)
    print(f"Found {len(training_samples)} categories in training set")

    print("Loading test data...")
    test_df = read_csv_with_encoding(TEST_CSV)
    print(f"Test data shape: {test_df.shape}")

    unknown_categories = set(test_df['categoryName']) - set(label_encoder.classes_)
    if unknown_categories:
        print(f"Warning: Found {len(unknown_categories)} categories in test set that were not in training set.")
        print("These categories will be skipped during evaluation.")
        test_df = test_df[~test_df['categoryName'].isin(unknown_categories)]

    test_df['label'] = label_encoder.transform(test_df['categoryName'])

    model = FusionModel(num_classes=num_classes).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")

    test_dataset = TestDataset(test_df, BASE_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0
    )

    class_predictions = {name: {'correct': 0, 'total': 0} for name in label_encoder.classes_}
    total_loss = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()

    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if not batch:
                continue

            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            categories = batch['category_name']

            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            for pred, label, category in zip(preds.cpu(), labels.cpu(), categories):
                class_predictions[category]['total'] += 1
                if pred == label:
                    class_predictions[category]['correct'] += 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    confusion = confusion_matrix(all_labels, all_preds, labels=range(num_classes)).tolist()

    min_test_samples = 10
    class_metrics = calculate_metrics(
        class_predictions,
        test_df['categoryName'].value_counts().to_dict(),
        training_samples,
        min_test_samples
    )

    top_200_classes = dict(sorted(
        class_metrics.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )[:200])

    results = {
        'overall_metrics': {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'macro_f1_score': macro_f1,
            'balanced_accuracy': balanced_accuracy,
            'total_test_samples': len(test_df),
            'total_classes_in_test': len(class_metrics),
            'min_test_samples_threshold': min_test_samples
        },
        'top_200_classes': top_200_classes,
        'class_order': label_encoder.classes_.tolist(),
        'confusion_matrix': confusion,
    }


    output_file = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    report_file = os.path.join(RESULTS_DIR, 'top_200_classes_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"Top 200 Categories (Minimum {min_test_samples} test samples required)\n")
        f.write("=" * 120 + "\n")
        header = f"{'Category Name':<50} {'Accuracy':<10} {'Train Samples':<15} {'Test Samples':<15} {'Score':<10}\n"
        f.write(header)
        f.write("-" * 120 + "\n")

        for category, metrics in top_200_classes.items():
            line = f"{category:<50} {metrics['accuracy']:<10.4f} {metrics['train_samples']:<15} "
            line += f"{metrics['test_samples']:<15} {metrics['score']:<10.4f}\n"
            f.write(line)

    print("\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Total test samples: {len(test_df)}")
    print(f"Classes with sufficient test samples (>={min_test_samples}): {len(class_metrics)}")
    print(f"\nDetailed results have been saved to:")
    print(f"1. JSON format: {output_file}")
    print(f"2. Text report: {report_file}")

if __name__ == '__main__':
    evaluate_model()
