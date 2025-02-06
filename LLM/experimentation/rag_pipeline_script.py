import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import LlamaForCausalLM, LlamaTokenizerFast       # LLM for report classificuing
from sentence_transformers import SentenceTransformer       # for embedding model
from sklearn.model_selection import train_test_split
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, pipeline#, AutoModelForSeq2SeqGeneration
# from transformers import AutoModelForSeq2SeqGeneration

import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

from transformers import AutoModelForSeq2SeqLM


filename = "data/labeled_data_combined_reports.csv"

# load data in df format
df_reports = pd.read_csv(filename)

df_reports["report_and_frac_label"] = (
    "Report:\n" + 
    df_reports["combined_reports"] + 
    "\n\nFracture classification:\n" + 
    df_reports["image_ct___1"]
    # df_reports["image_ct___1"].apply(lambda x: "Positive" if float(x) > 0 else "Negative")
)

df_reports["report_and_mets_label"] = (
    "Report:\n" + 
    df_reports["combined_reports"] + 
    "\n\nMetastases classification:\n" + 
    df_reports["image_ct___1"]
    # df_reports["image_ct___2"].apply(lambda x: "Positive" if float(x) > 0 else "Negative")
)

# drop reports that have NaN in reports column
df_reports = df_reports.dropna(subset=["report_and_frac_label", "report_and_mets_label"])


class RAGPipeline:
    def __init__(self, 
                 embedding_model_name="sentence-transformers/all-mpnet-base-v2",
                 llm_model_name="google/flan-t5-large",  # Example seq2seq model
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        print(f"Initializing RAG system on {device}")
        
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            trust_remote_code=True
        )
        
        # load LLM
        print("Loading language model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            load_in_8bit=True if device == "cuda" else False
        )
        
        # text gen pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            # device=device,
            do_sample=True
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        self.vectorstore = None
        self.qa_chain = None

    def _create_prompt(self, context, question):
        return f"""Answer the following question based on the given context.

Context: {context}

Question: {question}

Answer:"""

    def load_data(self, df, text_column):

        loader = DataFrameLoader(df, page_content_column=text_column)
        documents = loader.load()
        
        texts = self.text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
        )
        
        return f"Loaded {len(texts)} text chunks into the vector store"

    def query(self, question):
        """query RAG system"""
        if self.qa_chain is None:
            raise ValueError("Please load data first")
        
        # retrieve relevant docs
        docs = self.vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = self._create_prompt(context, question)
        
        response = self.llm(prompt)
        
        return response[0]['generated_text'] if isinstance(response, list) else response

    def batch_query(self, questions):
        """be able to handle multiple reports at once"""
        return [self.query(q) for q in questions]

    def similarity_search(self, query, k=3):
        if self.vectorstore is None:
            raise ValueError("Please load data first")
        return self.vectorstore.similarity_search(query, k=k)

    def save_vectorstore(self, path):
        """save FAISS vector database locally"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Vector database saved to {path}")
        else:
            raise ValueError("No vector store to save")

    def load_vectorstore(self, path):
        """Load a saved FAISS vector store"""
        print(f"Loading vector store from {path}")
        self.vectorstore = FAISS.load_local(path, self.embeddings)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 1}
            )
        )

### TESTING WITH REPORTS
rag = RAGPipeline(
    # embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    # llm_model_name="meta-llama/Llama-2-7b-chat-hf"  # Or any other causal LM
)

print("Loading data...")
result = rag.load_data(df_reports.head(100), text_column='report_and_frac_label')
print(result)

fracture_prompt = f"""
We need help classifying whether this radiology report indicates a fracture. 
Instructions:
- You will be given an input report. 
- Please respond with "Positive" if the report indicates a fracture, or "Negative" if it does not.
- Please ONLY say one word, "Positive" or "Negative".
- If you are unsure if a report indicates a fracture, say "Positive".
- Please use the provided examples to inform your decision.

Here is the report to classify\n\n:
"""

texts = df_reports['combined_reports'].tolist()
frac_labels = df_reports['image_ct___1'].astype(float).round().astype(int).values.tolist()
mets_labels = df_reports['image_ct___2'].astype(float).round().astype(int).values.tolist()

frac_train_texts, frac_test_texts, frac_train_labels, frac_test_labels = train_test_split(
    texts, frac_labels, test_size=0.2, random_state=42
)

mets_train_texts, mets_test_texts, mets_train_labels, mets_test_labels = train_test_split(
    texts, mets_labels, test_size=0.2, random_state=42
)

print("Generating predictions...")
predictions = []
true_labels = []

outputs = rag.batch_query(frac_test_texts)

for i, o in zip(inputs, outputs):
    print(i)
    # print(f"Input: {i}")
    # print(f"Output: {o}")


i = 0
for report, label in zip(frac_test_texts, frac_test_labels):
    print("Prediction: " + str(i))
    query = fracture_prompt + report
    prediction = rag.query(query)
    predictions.append(prediction == "Positive")
    true_labels.append(label > 0)
    # print("\nPredictions: " + str(predictions))
    # print("\nTrue labels: " + str(true_labels))
    i = i + 1
    if i == 20:
        break

predictions = np.array(predictions)
true_labels = np.array(true_labels)

print("Evaluating performance...")
accuracy = accuracy_score(valid_true_labels, valid_predictions)
f1 = f1_score(valid_true_labels, valid_predictions, average="binary")  # Adjust as needed
print("F1: ", f1)