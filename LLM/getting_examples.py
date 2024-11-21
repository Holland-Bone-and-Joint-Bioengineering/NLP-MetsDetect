from sklearn.model_selection import train_test_split
import pandas as pd


grouped_reports = pd.read_csv('labeled_data.csv')  # Replace with the correct path
texts = grouped_reports['reports'].tolist()
labels = grouped_reports[['image_ct___1', 'image_ct___2']].astype(int).values.tolist()

# Split into train/test sets (optional, if not already split)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

example1 = train_texts[0]
example1_label = train_labels[0]
print("first example")
print(example1)
print("\nfirst example label")
print(example1_label)


example2 = train_texts[1]
example2_label = train_labels[1]
print("second example")
print(example2)
print("\nsecond example label")
print(example2_label)

example3 = train_texts[2]
example3_label = train_labels[2]
print("third example")
print(example3)
print("\nthird example label")
print(example3_label)

example4 = train_texts[3]
example4_label = train_labels[3]
print("fourth example")
print(example4)
print("\nfourth example label")
print(example4_label)

example5 = train_texts[4]
example5_label = train_labels[4]
print("fifth example")
print(example5)
print("\nfifth example label")
print(example5_label)

example6 = train_texts[5]
example6_label = train_labels[5]
print("sixth example")
print(example6)
print("\nsixth example label")
print(example6_label)