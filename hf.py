# Adapted from https://huggingface.co/blog/ray-tune
import argparse

from datasets import load_dataset, load_metric
from dvclive.huggingface import DVCLiveCallback
import ray
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

parser = argparse.ArgumentParser()
parser.add_argument("--address")
args = parser.parse_args()
ray.init(address=args.address)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
dataset = load_dataset('glue', 'mrpc')
metric = load_metric('glue', 'mrpc')

def encode(examples):
    outputs = tokenizer(
        examples['sentence1'], examples['sentence2'], truncation=True)
    return outputs

encoded_dataset = dataset.map(encode, batched=True)

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', return_dict=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Evaluate during training and a bit more often
# than the default to be able to prune bad trials early.
# Disabling tqdm is a matter of preference.
training_args = TrainingArguments(
    "test", evaluation_strategy="steps", eval_steps=500, disable_tqdm=True)
trainer = Trainer(
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    model_init=model_init,
    compute_metrics=compute_metrics,
)
trainer.add_callback(DVCLiveCallback())

# trainer.train()
# Default objective is the sum of all metrics
# when metrics are provided, so we have to maximize it.
if args.address:
    trainer.hyperparameter_search(
        direction="maximize", 
        backend="ray", 
        n_trials=3, # number of trials
    )
else:
    trainer.hyperparameter_search(
        direction="maximize", 
        backend="ray", 
        n_trials=3, # number of trials
        resources_per_trial={"cpu": 1, "gpu": 0},
    )
