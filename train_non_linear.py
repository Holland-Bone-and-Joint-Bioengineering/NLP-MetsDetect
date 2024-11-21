import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification


# TODO:
'''
# Initialize the model
model_name = 'zzxslp/RadBERT-RoBERTa-4m'
num_labels = 2
model = ClassificationModelWrapper(model_name, num_labels)'''
class ClassificationModelWrapper(nn.Module):
    def __init__(self, model_name, num_labels):
        super(ClassificationModelWrapper, self).__init__()
        # Load the pre-trained model with a classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.sigmoid = nn.Sigmoid()  # Add sigmoid for binary classification

    def forward(self, input_ids, attention_mask):
        # Get the model's output
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Extract logits
        probs = self.sigmoid(logits)  # Apply sigmoid activation
        return probs  # Return probabilities

class EmbeddingsNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmbeddingsNN, self).__init__()
        
        # self.bn1 = nn.BatchNorm1d(input_dim)
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = self.bn1(x)
        x = self.mlp1(x)
        x = self.relu(x)
        # x = self.bn2(x)
        x = self.mlp2(x)
        return x

# Define the RadBERT Classifier
class RadBERTClassifier(nn.Module):
    def __init__(self, base_model, num_labels):
        super(RadBERTClassifier, self).__init__()
        self.base_model = base_model  # RadBERT or any other transformer model
        self.dropout = nn.Dropout(0.3)  # Regularization

        # Use EmbeddingsNN instead of simple linear classifier
        self.classifier = EmbeddingsNN(
            input_dim=base_model.config.hidden_size, 
            hidden_dim=64,
            output_dim=num_labels
        )

    def forward(self, input_ids, attention_mask):
        # Get the outputs from the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the `[CLS]` token's hidden state
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        
        # Apply dropout
        cls_output = self.dropout(cls_output)
        
        # Pass through the classification head
        logits = self.classifier(cls_output)  # Output logits for each label
        
        return torch.sigmoid(logits)  # Apply sigmoid for independent labels


# Define the Radiology Dataset
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

        # Tokenize the text
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


# Load the model, tokenizer, and configuration
config = AutoConfig.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
tokenizer = AutoTokenizer.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
base_model = AutoModel.from_pretrained('zzxslp/RadBERT-RoBERTa-4m', config=config)

# Load and prepare the data
grouped_reports = pd.read_csv('labeled_data.csv')
texts = grouped_reports['reports'].tolist()
labels = grouped_reports[['image_ct___1', 'image_ct___2']].astype(int).values.tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Create datasets and dataloaders
train_dataset = RadiologyDataset(train_texts, train_labels, tokenizer)
test_dataset = RadiologyDataset(test_texts, test_labels, tokenizer)

batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

model = RadBERTClassifier(base_model, 2)


# Set up training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Freeze base model parameters
'''
for param in model.base_model.parameters():
    param.requires_grad = False'''
for name, param in model.base_model.encoder.layer.named_parameters():
    layer_num = int(name.split('.')[0])  # Get the layer number
    if layer_num < 9:  # Adjust this number based on your needs
        param.requires_grad = False

criterion = nn.BCELoss()  # Binary cross-entropy loss
# Lower learning rate for BERT layers
bert_lr = 2e-5  

# Higher learning rate for MLP classifier head
mlp_lr = 1e-3  

# Parameter groups
optimizer = optim.AdamW([
    {'params': model.base_model.parameters(), 'lr': bert_lr},  # BERT parameters
    {'params': model.classifier.parameters(), 'lr': mlp_lr}    # MLP head parameters
])

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}")


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = (outputs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy and F1 score
    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_f1 = f1_score(all_labels, all_preds, average="micro")

    # Per-label accuracy
    num_labels = all_labels.shape[1]
    per_label_accuracy = [
        accuracy_score(all_labels[:, i], all_preds[:, i]) for i in range(num_labels)
    ]

    # Print the accuracy for each label
    for i, acc in enumerate(per_label_accuracy):
        print(f"Accuracy for Label {i + 1}: {acc:.4f}")

    return overall_accuracy, overall_f1, per_label_accuracy


# Evaluate the model
test_accuracy, test_f1, per_label_accuracies = evaluate_model(model, test_dataloader, device)
print(f"Overall Test Accuracy: {test_accuracy:.4f}")
print(f"Overall F1 Score: {test_f1:.4f}")
