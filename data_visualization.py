import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import nltk
import pandas as pd
from nltk import FreqDist
from nltk.util import ngrams


# Load the labeled data
grouped_reports = pd.read_csv('data/labeled_data.csv')

# Tokenize sentences
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

grouped_reports['tokens'] = grouped_reports['reports'].apply(nltk.word_tokenize)

# Generate bigrams (can be changed to trigrams, etc.)
n = 2
grouped_reports['bigrams'] = grouped_reports['tokens'].apply(lambda x: list(ngrams(x, n)))

# Flatten the list of bigrams
all_bigrams = [bigram for sublist in grouped_reports['bigrams'] for bigram in sublist]

# Frequency distribution of bigrams
fdist = FreqDist(all_bigrams)

# Create DataFrame for bigram frequencies
bigram_freq_df = pd.DataFrame(fdist.items(), columns=["Bigram", "Frequency"])
bigram_freq_df = bigram_freq_df.sort_values(by="Frequency", ascending=False)

# Show the result
print(bigram_freq_df)

# Phrase "[DEIDENTIFIED - DOCTOR'S INFO]" occurs frequently, remove before concatenating reports

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')

# Extract the report texts and labels
reports = grouped_reports['reports']
labels = grouped_reports[['image_ct___1', 'image_ct___2']]

# 1. Calculate the tokenized length of each report
tokenized_lengths = reports.apply(lambda x: len(tokenizer.encode(x, truncation=False)))

# Plot a histogram of tokenized report lengths
plt.figure(figsize=(10, 6))
plt.hist(tokenized_lengths, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(tokenized_lengths.mean(), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean Length: {tokenized_lengths.mean():.2f}')
plt.title('Distribution of Tokenized Report Lengths', fontsize=14)
plt.xlabel('Tokenized Report Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.savefig('tokenized_report_length_distribution.png', dpi=300)  # Save the plot
plt.show()

# 2. Calculate counts for the 2x2 label classes
labels['class'] = labels['image_ct___1'] * 2 + labels['image_ct___2']
class_counts = labels['class'].value_counts().sort_index()

# Map the classes to meaningful labels
class_mapping = {
    0: (0, 0),  # No Fractures, No Metastases
    1: (0, 1),  # No Fractures, Metastases
    2: (1, 0),  # Fractures, No Metastases
    3: (1, 1)   # Fractures, Metastases
}

# Prepare the 2x2 grid data
grid_data = [[0, 0], [0, 0]]  # Initialize a 2x2 grid
for label, count in class_counts.items():
    x, y = class_mapping[label]
    grid_data[x][y] = count

# Plot the 2x2 grid
plt.figure(figsize=(8, 8))
plt.imshow(grid_data, cmap='Blues', interpolation='nearest')
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(grid_data[i][j]), ha='center', va='center', color='black', fontsize=14)

plt.title('2x2 Label Class Distribution', fontsize=16)
plt.xticks([0, 1], ['No Metastases', 'Metastases'], fontsize=10)
plt.yticks([0, 1], ['No Fractures', 'Fractures'], fontsize=10)
plt.xlabel('Metastases Presence', fontsize=14)
plt.ylabel('Fracture Presence', fontsize=14)
plt.colorbar(label='Number of Samples')
plt.savefig('label_class_2x2_distribution.png', dpi=300)  # Save the plot
plt.show()

# Print mean tokenized length for reference
print(f"Mean Tokenized Report Length: {tokenized_lengths.mean():.2f}")
