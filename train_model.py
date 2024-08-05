# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast
from datasets import Dataset

# Charger le dataset
df = pd.read_csv('tweets.csv')

# Diviser le dataset en ensembles d'entraînement et de validation
train_df, val_df = train_test_split(df, test_size=0.2)

# Convertir les DataFrames en Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Charger le tokenizer et le modèle
tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=2)

# Prétraiter les données
def preprocess_function(examples):
    return tokenizer(examples['tweet'], truncation=True, padding=True)

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Entraîner le modèle
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Sauvegarder le modèle entraîné
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
