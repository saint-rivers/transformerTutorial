from gc import callbacks
from tokenize import tokenize

from datasets import load_dataset
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset = load_dataset("csebuetnlp/xlsum", "english")


# print(dataset['train'][2])
model_checkpoint = "t5-small"
from transformers import AutoTokenizer, BatchEncoding

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# this is what the model requires
# as it has many other functions other than summarize, like translate
prefix = "summarize: "


def preprocess_function(examples) -> BatchEncoding:
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True) # why does this have to be truncated?

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=1024, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = dataset.map(preprocess_function, batched=True)


from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")

train_dataset = tokenized_datasets["validation"].to_tf_dataset(
    batch_size=8,
    columns=["input_ids", "attention_mask", "labels"],
    shuffle=True,
    collate_fn=data_collator
)


validation_dataset = tokenized_datasets["test"].to_tf_dataset(
    batch_size=8,
    columns=["input_ids", "attention_mask", "labels"],
    shuffle=False,
    collate_fn=data_collator
)


from transformers import AdamWeightDecay
import tensorflow as tf

optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)

from huggingface_hub import notebook_login
notebook_login()

from transformers.keras_callbacks import PushToHubCallback

callback = PushToHubCallback(
    output_dir="saintrivers/summarization-tutorial", tokenizer=tokenizer, save_strategy="epoch"
)

model.fit(
    train_dataset, validation_data=validation_dataset, callbacks=[callback], epochs=3
)









