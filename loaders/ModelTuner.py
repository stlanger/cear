import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))

from tqdm import tqdm
from datasets import Dataset

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

from transformers import AutoTokenizer
from transformers import create_optimizer
from transformers import TFAutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers.keras_callbacks import KerasMetricCallback
import evaluate

from model_saver_callback import ModelSaverCallback


class ModelTuner:

    input_ids = []
    labels = []
    attention_mask = []

    tokenizer = None
    model = None    
    
    def __init__(self, model_name, label_list, id2label, label2id):
        self.model_name = model_name

        self.label_list = label_list
        self.id2label = id2label
        self.label2id = label2id

        self.html_elems = {
            1: ("<b style='color:red; font-size:1.2em;'><i>", "</i></b>"),
            3: ("<b style='color:blue; font-size:1.5em;'><i>","</i></b>")    
        }

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def align_token_data(self, spans, labels):        
        for s, l in zip(spans, labels):
            tokenized = self.tokenizer(s, truncation=True, is_split_into_words=True)        
            self.labels.append(self.__align_labels(tokenized, l))
            self.input_ids.append(tokenized["input_ids"])
            self.attention_mask.append(tokenized["attention_mask"])


    def load_model(self, model_path):
        self.model = TFAutoModelForTokenClassification.from_pretrained(
            model_path, num_labels=len(self.label_list), id2label=self.id2label, label2id=self.label2id
        )
    

    def train(self, train_size=0.9, num_train_epochs=30, batch_size=16, init_lr=2e-5, weight_decay=0.01, num_warmup_steps=0):
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, return_tensors="tf")
        ds = Dataset.from_dict({
                "input_ids": self.input_ids,
                "labels": self.labels,
                "attention_mask": self.attention_mask
            })
        ds = ds.train_test_split(train_size=train_size, seed=42, shuffle=True)

        num_train_steps = (len(self.input_ids) // batch_size) * num_train_epochs
        print("num_train_steps:", num_train_steps)
        optimizer, lr_schedule = create_optimizer(
            init_lr = init_lr,
            num_train_steps = num_train_steps,
            weight_decay_rate = weight_decay,
            num_warmup_steps = num_warmup_steps,
        )
        
        self.model = TFAutoModelForTokenClassification.from_pretrained(
            self.model_name, num_labels=len(self.label_list), id2label=self.id2label, label2id=self.label2id
        )        
        tf_train_set = self.model.prepare_tf_dataset(
            ds["train"],
            shuffle = True,
            batch_size = batch_size,
            collate_fn = data_collator,
        )        
        tf_validation_set = self.model.prepare_tf_dataset(
            ds["test"],
            shuffle = True,
            batch_size = batch_size,
            collate_fn = data_collator,
        )           
        self.model.compile(optimizer=optimizer)  # No loss argument!        
        metric_callback = KerasMetricCallback(metric_fn=self.__compute_metrics, eval_dataset=tf_validation_set)

        save_model = ModelSaverCallback(self.model, folder="./model", filename=f"chemical_extract_{self.model_name.replace('/', '-')}", save_each_epoch=False)
        self.model.fit(x=tf_train_set, epochs = num_train_epochs, callbacks=[metric_callback, save_model])
        self.model.summary()

    
    def infer(self, text):
        specials = []
            
        tokenized = self.tokenizer(text, return_tensors="tf")
        offset_mapping = self.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]    
        
        logits = self.model(**tokenized).logits
        predicted_probs = tf.nn.softmax(logits, axis=2)    
        
        predicted_token_class_ids = tf.math.argmax(logits, axis=-1)[0]  
        predicted_token_class = [self.model.config.id2label[t] for t in predicted_token_class_ids.numpy().tolist()]
        probs = tf.math.reduce_max(predicted_probs, axis=-1)
    
        # for l, t in zip(predicted_token_class_ids, tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])):        
        #     print(id2label[int(l)], t)
    
        label, start, end = 0, 0, 0    
    
        i = 0
        while i < len(predicted_token_class_ids):
            l = predicted_token_class_ids[i]
            if l % 2 == 1:
                label = int(l)
                start = offset_mapping[i][0]
                while i+1 < len(predicted_token_class_ids) and predicted_token_class_ids[i + 1] == l+1:
                    i += 1
                end = offset_mapping[i][1]
                specials.append((label, start, end))
            i += 1
        return specials
    
    # return spans with start and end index and a class
    def infer_html(self, text):
        return self.__render_as_html(text, self.infer(text))


    def __render_as_html(self, text, specials):        
        start_dict = {}
        end_dict = {}
        for s in specials:
            start_dict[s[1]] = self.html_elems[s[0]][0]
            end_dict[s[2]] = self.html_elems[s[0]][1]
        html = ""    
        for i, c  in enumerate(text):                
            start_elem = start_dict.get(i, None)
            end_elem = end_dict.get(i, None)
    
            if start_elem:
                html += start_elem
            if end_elem:
                html += end_elem
            html += c
        return html
       
    
    def __compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        seqeval = evaluate.load("seqeval") 
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        
    
    def __align_labels(self, tokenized_inputs, labels):
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(labels[word_idx])
            else:
                previous_label = labels[previous_word_idx]
                if (previous_label > 0):
                    label_ids.append(previous_label + 1)
                else:
                    label_ids.append(0)
            previous_word_idx = word_idx
        return label_ids


    
    