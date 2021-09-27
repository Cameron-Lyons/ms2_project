import pandas as pd
import numpy as np
import textstat
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

df = pd.read_csv("data/WikiLarge_train.csv")
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
X_train["clean_text"] = X_train["original_text"].str.replace("[^\w\s]", "", regex=True)
X_train["clean_text"] = X_train["clean_text"].str.replace("\s+", " ", regex=True)

X_train["flesch"] = X_train["clean_text"].apply(textstat.flesch_reading_ease)
X_train["flesch_kincaid"] = X_train["clean_text"].apply(textstat.flesch_kincaid_grade)
X_train["gunning_fog"] = X_train["clean_text"].apply(textstat.gunning_fog)
X_train["readability"] = X_train["clean_text"].apply(textstat.automated_readability_index)
X_train["coleman_liau"] = X_train["clean_text"].apply(textstat.coleman_liau_index)
X_train["linsear_write"] = X_train["clean_text"].apply(textstat.linsear_write_formula)
X_train["dale_chall"] = X_train["clean_text"].apply(textstat.dale_chall_readability_score)
X_train["spache"] = X_train["clean_text"].apply(textstat.spache_readability)
X_train["difficult_words"] = X_train["clean_text"].apply(textstat.difficult_words)

X_test["flesch"] = X_test["clean_text"].apply(textstat.flesch_reading_ease)
X_test["flesch_kincaid"] = X_test["clean_text"].apply(textstat.flesch_kincaid_grade)
X_test["gunning_fog"] = X_test["clean_text"].apply(textstat.gunning_fog)
X_test["readability"] = X_test["clean_text"].apply(textstat.automated_readability_index)
X_test["coleman_liau"] = X_test["clean_text"].apply(textstat.coleman_liau_index)
X_test["linsear_write"] = X_test["clean_text"].apply(textstat.linsear_write_formula)
X_test["dale_chall"] = X_test["clean_text"].apply(textstat.dale_chall_readability_score)
X_test["spache"] = X_test["clean_text"].apply(textstat.spache_readability)
X_test["difficult_words"] = X_train["clean_text"].apply(textstat.difficult_words)

X_train['char_count'] = X_train['clean_text'].apply(len)
X_train['word_count'] = X_train['clean_text'].apply(lambda x: len(x.split()))
X_train['word_density'] = X_train['char_count'] / (X_train['word_count']+1)

X_test['char_count'] = X_test['clean_text'].apply(len)
X_test['word_count'] = X_test['clean_text'].apply(lambda x: len(x.split()))
X_test['word_density'] = X_test['char_count'] / (X_test['word_count']+1)

vectorizer = TfidfVectorizer()
train_tfidf_vec = vectorizer.fit_transform(X_train["clean_text"])
test_tfidf_vec = vectorizer.fit_transform(X_test["clean_text"])

nb = MultinomialNB()
nb.fit(train_tfidf_vec, y_train.to_numpy())
X_train["nb_preds"] = nb.predict(train_tfidf_vec)
X_test["nb_preds"] = nb.predict(test_tfidf_vec)


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(X_train["clean_text"].to_numpy(), truncation=True, padding=True)
test_encodings = tokenizer(X_test["clean_text"].to_numpy(), truncation=True, padding=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

class wikidataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = wikidataset(train_encodings, y_train.to_numpy())
test_dataset = wikidataset(test_encodings, y_test.to_numpy())
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset
)

trainer.train()

train_preds = []
test_preds = []
with torch.no_grad():
    
    for i in range(len(X_train)):
        text = train_dataset[i]
        
        prediction = model(text)
        
        predicted_class = np.argmax(prediction)
        train_preds.append(predicted_class)
    
    for i in range(len(X_test)):
        text = train_dataset[i]
        
        prediction = model(text)
        
        predicted_class = np.argmax(prediction)
        test_preds.append(predicted_class)

X_train["bert_preds"] = train_preds
X_test["bert_preds"] = test_preds

dale_chall = open("data/dale_chall.txt", "r")

dc_words = []

for word in dale_chall:
  dc_words.append(word)

dale_chall.close()

def n_difficult_words(text):
    difficult_words = set([word for word in text.split() if word not in dc_words])
    return len(difficult_words)

X_train["n_difficult_words"] = X_train["clean_text"].apply(n_difficult_words)
X_test["n_difficult_words"] = X_test["clean_text"].apply(n_difficult_words)

aoa = pd.read_csv("data/AoA_51715_words.csv")

aoa = dict(zip(aoa.Word, aoa.AoA_Kup))

def aoa_max_avg(text):
    aoas = [aoa[word] for word in text.split()]
    return max(aoas), sum(aoas) / len(aoas)

X_train["max_aoa"], X_train["avg_aoa"] = X_train["clean_text"].apply(aoa_max_avg)
X_test["max_aoa"], X_test["avg_aoa"] = X_test["clean_text"].apply(aoa_max_avg)

features = ["flesch",
            "flesch_kincaid",
            "gunning_fog",
            "readability",
            "coleman_liau",
            "linsear_write",
            "dale_chall",
            "spache",
            "char_count",
            "word_count",
            "nb_preds",
            "bert_preds",
            "n_difficult_words",
            "max_aoa",
            "avg_aoa"]

clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)
print(accuracy_score(y_test, y_preds))
