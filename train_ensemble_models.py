import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

class EmbeddingsNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.mlp2(x)
        return x

class RadBERTClassifier(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = EmbeddingsNN(
            input_dim=base_model.config.hidden_size,
            hidden_dim=64,
            output_dim=num_labels
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return torch.sigmoid(logits)

class RadiologyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        labels = self.labels[index]

        encoded = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

class NaiveBayesWrapper:
    def __init__(self, num_labels):
        self.vectorizer = CountVectorizer()
        self.classifiers = [MultinomialNB() for _ in range(num_labels)]
    
    def fit(self, texts, labels):
        if isinstance(labels, list):
            labels = np.array(labels)

        X = self.vectorizer.fit_transform(texts)
        for i, clf in enumerate(self.classifiers):
            clf.fit(X, labels[:, i])
    
    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        return np.array([clf.predict_proba(X)[:, 1] for clf in self.classifiers]).T

class RadiologyEnsemble:
    def __init__(self, model_name='zzxslp/RadBERT-RoBERTa-4m', num_labels=2, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        
        # Initialize RadBERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModel.from_pretrained(model_name, config=self.config)
        self.radbert_model = RadBERTClassifier(base_model, num_labels).to(self.device)
        
        # Initialize Naive Bayes
        self.nb_model = NaiveBayesWrapper(num_labels)
        
        # Initialize weights for ensemble (can be adjusted)
        self.weights = {'radbert': 0.7, 'nb': 0.3}

        for name, param in self.radbert_model.base_model.encoder.layer.named_parameters():
            layer_num = int(name.split('.')[0])  # Get the layer number
            if layer_num < 9:  # Adjust this number based on your needs
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def train_radbert(self, train_dataloader, epochs=12, bert_lr=2e-5, mlp_lr=1e-3):
        criterion = nn.BCELoss()
        optimizer = optim.AdamW([
            {'params': self.radbert_model.base_model.parameters(), 'lr': bert_lr},
            {'params': self.radbert_model.classifier.parameters(), 'lr': mlp_lr}
        ])

        for epoch in range(epochs):
            self.radbert_model.train()
            train_loss = 0

            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.radbert_model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_loss = train_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}")
    
    def train_nb(self, texts, labels):
        self.nb_model.fit(texts, labels)
    
    def get_radbert_predictions(self, dataloader):
        self.radbert_model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.radbert_model(input_ids=input_ids, attention_mask=attention_mask)
                all_preds.extend(outputs.cpu().numpy())
        
        return np.array(all_preds)
    
    def get_ensemble_predictions(self, texts, dataloader):
        # Get predictions from both models
        radbert_preds = self.get_radbert_predictions(dataloader)
        nb_preds = self.nb_model.predict_proba(texts)
        
        # Weighted average of predictions
        ensemble_preds = (
            self.weights['radbert'] * radbert_preds + 
            self.weights['nb'] * nb_preds
        )
        
        # Convert to binary predictions
        return (ensemble_preds > 0.5).astype(int)
    
    def evaluate(self, texts, dataloader, true_labels):
        if isinstance(true_labels, list):
            true_labels = np.array(true_labels)
        predictions = self.get_ensemble_predictions(texts, dataloader)
        
        # Calculate metrics
        overall_accuracy = accuracy_score(true_labels, predictions)
        overall_f1 = f1_score(true_labels, predictions, average='micro')
        
        # Per-label accuracy
        per_label_accuracy = [
            accuracy_score(true_labels[:, i], predictions[:, i])
            for i in range(self.num_labels)
        ]
        
        return {
            'accuracy': overall_accuracy,
            'f1': overall_f1,
            'per_label_accuracy': per_label_accuracy
        }

# Example usage
def main():
    # Load data
    grouped_reports = pd.read_csv('labeled_data.csv')
    texts = grouped_reports['reports'].tolist()
    labels = grouped_reports[['image_ct___1', 'image_ct___2']].astype(int).values.tolist()

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Initialize ensemble
    ensemble = RadiologyEnsemble()

    # Create dataset and dataloader for RadBERT
    train_dataset = RadiologyDataset(train_texts, train_labels, ensemble.tokenizer)
    test_dataset = RadiologyDataset(test_texts, test_labels, ensemble.tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    # Train both models
    print("Training RadBERT...")
    ensemble.train_radbert(train_dataloader)
    
    print("Training Naive Bayes...")
    ensemble.train_nb(train_texts, train_labels)

    # Evaluate ensemble
    print("\nEvaluating Ensemble Model...")
    metrics = ensemble.evaluate(test_texts, test_dataloader, test_labels)
    
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Overall F1 Score: {metrics['f1']:.4f}")
    for i, acc in enumerate(metrics['per_label_accuracy']):
        print(f"Accuracy for Label {i + 1}: {acc:.4f}")

if __name__ == "__main__":
    main()