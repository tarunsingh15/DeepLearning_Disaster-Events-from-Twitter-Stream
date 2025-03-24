#%%
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import openai
# !pip install httpx==0.27.2 --force-reinstall
# !pip install --upgrade openai
#%%
random_state = 5814
#%%
data = pd.read_csv('train.csv')
#%%
def preprocessing(text):
    if pd.isna(text):
        return ''
    text = text.lower()
    text  = re.sub("#\S*\s", "", text)
    text  = re.sub("W+", "", text)
    text  = re.sub("@\S*\s", "", text)
    text  = re.sub("http\S*\s", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
#%%
data['text'] = data['text'].apply(preprocessing)
#%%
x_train, x_test, y_train, y_test = train_test_split(data['text'], data['target'],test_size=0.2, random_state=random_state)
#%% md
# # Traditional Models
#%%
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
#%%
tfidf = TfidfVectorizer(max_features=1000)
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)
#%%
traditional_models = {
    'LogisticRegression' : LogisticRegression(max_iter=1000,random_state=random_state),
    'naive_bayes' : MultinomialNB(),
    'svm' : LinearSVC(max_iter=1000,random_state=random_state),
}
#%%
traditional_results = {}
#%%
for name, model in traditional_models.items():
    model.fit(x_train_tfidf, y_train)
    y_pred = model.predict(x_test_tfidf)
    traditional_results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
#%%
print(traditional_results)
#%% md
# # BERT Model
#%%
model_name = 'bert-base-uncased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2,hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1).to(device)
#%%
def prep_data_bert(texts, labels):
    texts = texts.astype('str').tolist()
    encodings = tokenizer(texts, padding=True, truncation=True, max_length = 128,batch_size =12,return_tensors='pt')
    if labels is not None:
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels.values))
    else:
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])

    return DataLoader(dataset, batch_size=16, shuffle=True)
#%%
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
#%%
def train_bert(train_loader, epochs = 3):
    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-5, weight_decay = 0.01)
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        bert_model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = bert_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), max_norm=1.0)
            optimizer.step()
#%%
def bert_predict(test_loader):
    bert_model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.logits.argmax(dim=1).cpu().numpy())

    return predictions
#%%
train_loader = prep_data_bert(x_train, y_train)
test_loader = prep_data_bert(x_test,y_test)
#%%
train_bert(train_loader)
#%%
bert_predictions = bert_predict(test_loader)
#%% md
# # GPT Implementation
#%%
sample_size = 4
api_key = "<api_key_gpt>"
#%%
import openai
import time
import json

def gpt_predictions(texts, api_key, few_shot_samples=16):
    openai.api_key = api_key
    predictions = []

    template = """You are a highly trained assistant tasked with
            classifying tweets to determine if they are about a disaster or not.

        Your job is to read each tweet carefully and provide a classification as either:
        - 1 if the tweet is about a disaster
        - 0 if the tweet is not about a disaster

        For each classification,
        output the classification as a number (either 0 or 1),
        without any additional text or explanation.

        Here are some examples to help you understand the task:
        Tweet: 'The storm has caused severe flooding in the city, with many homes submerged.',
        Classification: 1
        Tweet: 'I love spending time with my friends at the beach.',
        Classification: 0
        Tweet: 'Earthquake tremors were felt throughout the region, with significant damage reported.',
        Classification: 1
        Tweet: 'It’s a beautiful day for a walk in the park!',
        Classification: 0

        Now, please classify the following tweet:
        Tweet: {tweet}, Classification:
    """

    for text in texts:
        formatted_prompt = template.format(tweet=text)
        print(f"Formatted Prompt for Classification:\n{formatted_prompt}")

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that classifies tweets as related to disasters or not."},
                {"role": "user", "content": formatted_prompt},
            ],
            max_tokens=10,
            temperature=0.5
        )

        try:
            prediction_text = response.choices[0].message.content.strip()
            print(f"Prediction for tweet '{text}': {prediction_text}")
            if prediction_text.isdigit():
                prediction = int(prediction_text)
            else:
                raise ValueError("Invalid response format")
            predictions.append(prediction)
        except (ValueError, KeyError) as e:
            print(f"Error processing response for '{text}': {e}")
            predictions.append(0)

        # to avoid hitting rate limits
        time.sleep(1)

    return predictions

#%%
sample_ids = np.random.choice(len(x_test), sample_size, replace=False)
x_test_samples = x_test.iloc[sample_ids]
#%%
print(x_test_samples)
#%%
predictions = gpt_predictions(x_test_samples, api_key=api_key)
print(predictions)
#%% md
# # Model Comparisons
#%%
def eval_models(y_true, y_pred, model):
    return {
        'Model' : model,
        'Precision' : precision_score(y_true, y_pred),
        'Recall' : recall_score(y_true, y_pred),
        'F1' : f1_score(y_true, y_pred)
    }
#%%
results = []
for name, metrics in traditional_models.items():
    y_pred = model.predict(x_test_tfidf)

    results.append({
        'Model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    })
print(results)
#%%
print(eval_models(y_test, bert_predictions[:len(y_test)], 'BERT'))
#%%
y_test_samples = y_test.iloc[sample_ids]
#%%
print(eval_models(y_test_samples, gpt_predictions(x_test_samples, api_key = api_key), 'GPT'))
