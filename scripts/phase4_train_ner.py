"""Bootstrap weak BIO labels and fine-tune DistilBERT with held-out metrics."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from datasets import Dataset
from seqeval.metrics import precision_score, recall_score, f1_score
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baseline_extractor import SKILL_RE  # noqa: E402
from document_processing import extract_document  # noqa: E402

MODEL_NAME = "distilbert/distilbert-base-uncased"
LABELS = ["O", "B-SKILL", "I-SKILL"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def bootstrap_record(path: Path) -> dict[str, list]:
    text = extract_document(path).text
    tokens = text.split()
    labels = ["O"] * len(tokens)
    for match in SKILL_RE.finditer(text):
        start_token = len(text[:match.start()].split())
        end_token = len(text[:match.end()].split())
        if start_token < len(tokens):
            labels[start_token] = "B-SKILL"
            for index in range(start_token + 1, min(end_token, len(tokens))):
                labels[index] = "I-SKILL"
    return {"tokens": tokens, "ner_tags": [LABEL_TO_ID[label] for label in labels]}


def main() -> None:
    resume_paths = sorted((ROOT / "data" / "synthetic" / "bootstrapped_resumes").glob("resume_*.docx"))
    records = [bootstrap_record(path) for path in resume_paths]
    if len(records) < 2:
        raise RuntimeError("At least two labeled documents are required for a held-out evaluation.")
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch: dict) -> dict:
        encoded = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True)
        all_labels = []
        for row, word_ids in enumerate(encoded.word_ids(batch_index=index) for index in range(len(batch["tokens"]))):
            previous = None
            labels = []
            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)
                elif word_id != previous:
                    labels.append(batch["ner_tags"][row][word_id])
                else:
                    labels.append(-100)
                previous = word_id
            all_labels.append(labels)
        encoded["labels"] = all_labels
        return encoded

    tokenized = split.map(tokenize, batched=True, remove_columns=dataset.column_names)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID)
    output_dir = ROOT / "artifacts" / "ner_distilbert"
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            num_train_epochs=3,
            eval_strategy="epoch",
            save_strategy="no",
            report_to=[],
            seed=42,
        ),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    training_started = time.perf_counter()
    trainer.train()
    training_seconds = round(time.perf_counter() - training_started, 2)
    predictions = trainer.predict(tokenized["test"])
    predicted_ids = predictions.predictions.argmax(axis=-1)
    true_sequences = []
    predicted_sequences = []
    for predicted, labels in zip(predicted_ids, tokenized["test"]["labels"]):
        true_sequences.append([LABELS[label] for label in labels if label != -100])
        predicted_sequences.append([LABELS[label] for label, original in zip(predicted, labels) if original != -100])
    metrics = {
        "data_status": "synthetic_bootstrapped_labels",
        "model": MODEL_NAME,
        "train_documents": len(split["train"]),
        "held_out_documents": len(split["test"]),
        "precision": precision_score(true_sequences, predicted_sequences),
        "recall": recall_score(true_sequences, predicted_sequences),
        "f1": f1_score(true_sequences, predicted_sequences),
        "training_seconds_cpu": training_seconds,
        "artifact": str(output_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "phase4_ner_real_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
