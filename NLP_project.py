import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import numpy as np
from datetime import datetime
import traceback
import random
import re
import json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import sacrebleu
import pyarabic.araby as araby
required_libraries = {
    'transformers': True,
    'sentencepiece': True,
    'torch': True,
    'datasets': True,
    'sacrebleu': True,
    'pyarabic': True
}
TRANSLATION_AVAILABLE = required_libraries['transformers'] and required_libraries['sentencepiece']
TRAINING_AVAILABLE = TRANSLATION_AVAILABLE and required_libraries['torch'] and required_libraries['datasets']
ARABIC_PREPROCESSING = required_libraries['pyarabic']
BLEU_AVAILABLE = required_libraries['sacrebleu']
random_seed = 123
BATCH_SIZE = 32
EPOCHS = 5
TEST_SPLIT = 0.2
DEFAULT_DATASET_PATH = r"C:\Users\Adham\Downloads\nlp_2\ara_eng.txt"
DEFAULT_MODEL_PATH = r"Helsinki-NLP/opus-mt-en-ar"
DEFAULT_SAVED_MODEL_PATH = r"./fine_tuned_model"
random.seed(random_seed)
def clean_english_text(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.,;:!?\'"-]', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    return text.strip()

def clean_arabic_text(text):
    if not text:
        return ""

    if ARABIC_PREPROCESSING:
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
        text = araby.normalize_hamza(text)
        text = araby.normalize_alef(text)
    else:
        text = re.sub(r'ـ', '', text)
        text = re.sub(r'[أإآ]', 'ا', text)

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    return text.strip()

def preprocess_data_pair(en_text, ar_text, apply_cleaning=True):
    if apply_cleaning:
        en_text = clean_english_text(en_text)
        ar_text = clean_arabic_text(ar_text)

    return en_text, ar_text

def load_and_preprocess_dataset(file_path=None, apply_cleaning=True, max_length=None):
    try:
        if not file_path or not os.path.exists(file_path):
            return None, f"Dataset file not found: {file_path}"

        data_pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                en_text = lines[i].strip()
                ar_text = lines[i + 1].strip()
                if en_text and ar_text:
                    en_text, ar_text = preprocess_data_pair(en_text, ar_text, apply_cleaning)
                    if max_length and (len(en_text.split()) > max_length or len(ar_text.split()) > max_length):
                        continue

                    data_pairs.append((en_text, ar_text))

        return data_pairs, f"Successfully loaded {len(data_pairs)} translation pairs"
    except Exception as e:
        traceback.print_exc()
        return None, f"Error loading dataset: {str(e)}"

def translate_text_en_to_ar(text, model_path=DEFAULT_MODEL_PATH):
    try:
        print(f"Translating: '{text}'")
        if not TRANSLATION_AVAILABLE:
            return "Translation libraries not available. Please install required packages."

        try:
            tokenizer = MarianTokenizer.from_pretrained(model_path)
            model = MarianMTModel.from_pretrained(model_path)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated = model.generate(**inputs)
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
            print(f"Translation result: '{result}'")
        except Exception as e:
            print(f"Detailed error using model: {str(e)}")
            print(traceback.format_exc())
            result = f"Unable to translate to Arabic. Error: {str(e)}\n\nTo fix this issue, please install the required libraries:\npip install transformers torch sentencepiece"
        return result
    except Exception as e:
        print(f"Translation error details: {str(e)}")
        print(traceback.format_exc())
        return f"Translation error: {str(e)}"

def calculate_bleu_score(references, hypothesis):
    if not BLEU_AVAILABLE:
        return None

    try:
        bleu = sacrebleu.corpus_bleu([hypothesis], [[references]])
        return bleu.score
    except Exception as e:
        print(f"Error calculating BLEU score: {str(e)}")
        return None

def calculate_mae(str1, str2):
    length_diff = abs(len(str1) - len(str2))
    min_len = min(len(str1), len(str2))
    char_diffs = sum(1 for i in range(min_len) if str1[i] != str2[i])
    total_error = length_diff + char_diffs
    return total_error / max(len(str1), len(str2)) if max(len(str1), len(str2)) > 0 else 0

def calculate_mse(str1, str2):
    min_len = min(len(str1), len(str2))
    max_len = max(len(str1), len(str2))
    squared_diffs = 0
    for i in range(min_len):
        diff = 1 if str1[i] != str2[i] else 0
        squared_diffs += diff ** 2

    for i in range(min_len, max_len):
        squared_diffs += 1

    return squared_diffs / max_len if max_len > 0 else 0

def calculate_accuracy(str1, str2):
    min_len = min(len(str1), len(str2))
    if min_len == 0:
        return 0.0

    correct_chars = 0
    for i in range(min_len):
        if str1[i] == str2[i]:
            correct_chars += 1

    return correct_chars / max(len(str1), len(str2))

def evaluate_translation(translations):
    if not translations:
        return {
            'avg_length': 0,
            'char_ratio': 0,
            'examples': []
        }
    total_source_length = sum(len(pair[0]) for pair in translations)
    total_target_length = sum(len(pair[1]) for pair in translations)
    avg_source_length = total_source_length / len(translations)
    avg_target_length = total_target_length / len(translations)
    char_ratio = total_target_length / total_source_length if total_source_length > 0 else 0
    mse_values = []
    mae_values = []
    accuracy_values = []
    bleu_scores = []
    for source, target in translations:
        mse = calculate_mse(source, target)
        mae = calculate_mae(source, target)
        accuracy = calculate_accuracy(source, target)
        mse_values.append(mse)
        mae_values.append(mae)
        accuracy_values.append(accuracy)
        if BLEU_AVAILABLE:
            bleu = calculate_bleu_score(source, target)
            if bleu is not None:
                bleu_scores.append(bleu)

    evaluation_results = {
        'avg_source_length': avg_source_length,
        'avg_target_length': avg_target_length,
        'char_ratio': char_ratio,
        'mse': np.mean(mse_values) if mse_values else 0,
        'mae': np.mean(mae_values) if mae_values else 0,
        'accuracy': np.mean(accuracy_values) if accuracy_values else 0,
        'examples': translations[:5]
    }
    if BLEU_AVAILABLE and bleu_scores:
        evaluation_results['bleu'] = np.mean(bleu_scores)

    return evaluation_results
##################################################################################################################################################################
# --- MODEL TRAINING FUNCTIONS ---
###################################
class TranslationDataset:
    def __init__(self, data_pairs, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_texts = [pair[0] for pair in data_pairs]
        self.target_texts = [pair[1] for pair in data_pairs]

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source = self.source_texts[idx]
        target = self.target_texts[idx]
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        source_ids = source_encoding["input_ids"].squeeze()
        target_ids = target_encoding["input_ids"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": target_ids
        }

def prepare_training_datasets(data_pairs, tokenizer, test_split=0.2, max_length=128):
    if not data_pairs:
        return None, None

    random.shuffle(data_pairs)
    split_idx = int(len(data_pairs) * (1 - test_split))
    train_pairs = data_pairs[:split_idx]
    test_pairs = data_pairs[split_idx:]
    train_source_texts = [pair[0] for pair in train_pairs]
    train_target_texts = [pair[1] for pair in train_pairs]
    test_source_texts = [pair[0] for pair in test_pairs]
    test_target_texts = [pair[1] for pair in test_pairs]
    train_dataset = Dataset.from_dict({
        "source": train_source_texts,
        "target": train_target_texts
    })
    test_dataset = Dataset.from_dict({
        "source": test_source_texts,
        "target": test_target_texts
    })
    def tokenize_function(examples):
        source_tokenized = tokenizer(
            examples["source"],
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        target_tokenized = tokenizer(
            examples["target"],
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        examples["input_ids"] = source_tokenized["input_ids"]
        examples["attention_mask"] = source_tokenized["attention_mask"]
        examples["labels"] = target_tokenized["input_ids"]
        return examples

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return train_dataset, test_dataset

def fine_tune_model(data_pairs, model_path=DEFAULT_MODEL_PATH, output_dir=DEFAULT_SAVED_MODEL_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=5e-5, progress_callback=None):
    if not TRAINING_AVAILABLE:
        return False, "Training libraries not available. Please install required packages."

    if not data_pairs or len(data_pairs) < 10:
        return False, "Insufficient data for fine-tuning. Please provide more examples."

    try:
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        model = MarianMTModel.from_pretrained(model_path)
        train_dataset, eval_dataset = prepare_training_datasets(
            data_pairs, tokenizer, test_split=TEST_SPLIT, max_length=128
        )
        if progress_callback:
            progress_callback("Datasets prepared. Starting training...")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            learning_rate=learning_rate,
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            load_best_model_at_end=True,
            report_to="none"
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )
        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        training_info = {
            "base_model": model_path,
            "dataset_size": len(data_pairs),
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(f"{output_dir}/training_info.json", "w", encoding="utf-8") as f:
            json.dump(training_info, f, indent=4)

        return True, f"Model fine-tuned successfully and saved to {output_dir}"
    except Exception as e:
        traceback.print_exc()
        return False, f"Error during fine-tuning: {str(e)}"
#############################################################################################################################################################################################################################################
# --- GUI APPLICATION ---
###########################
class EnhancedTranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced English-Arabic Translation")
        self.root.geometry("1200x900")
        self.loading_window = None
        self.evaluation_results = None
        self.evaluation_timestamp = None
        self.translations = []
        self.dataset = []
        self.dataset_path = DEFAULT_DATASET_PATH
        self.model_path = DEFAULT_MODEL_PATH
        self.saved_model_path = DEFAULT_SAVED_MODEL_PATH
        missing_libraries = [lib for lib, available in required_libraries.items() if not available]
        if missing_libraries:
            messagebox.showwarning(
                "Missing Libraries",
                f"Some features may be limited. Missing libraries: {', '.join(missing_libraries)}\n\n"
                f"Install with: pip install {' '.join(missing_libraries)}"
            )

        self.load_dataset()
        self.create_widgets()

    def load_dataset(self):
        if os.path.exists(self.dataset_path):
            self.dataset, message = load_and_preprocess_dataset(self.dataset_path)
            if not self.dataset:
                messagebox.showwarning("Dataset Warning", message)
        else:
            messagebox.showwarning(
                "Dataset Not Found",
                f"The dataset file was not found at: {self.dataset_path}\n"
                "You can still use the application, but some features may be limited."
            )

    def create_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.translation_tab = ttk.Frame(self.notebook)
        self.preprocessing_tab = ttk.Frame(self.notebook)
        self.training_tab = ttk.Frame(self.notebook)
        self.evaluation_tab = ttk.Frame(self.notebook)
        self.metrics_tab = ttk.Frame(self.notebook)
        self.dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.translation_tab, text="Translation")
        self.notebook.add(self.preprocessing_tab, text="Preprocessing")
        self.notebook.add(self.training_tab, text="Training")
        self.notebook.add(self.evaluation_tab, text="Evaluation")
        self.notebook.add(self.metrics_tab, text="Metrics")
        self.notebook.add(self.dataset_tab, text="Dataset")
        self.setup_translation_tab()
        self.setup_preprocessing_tab()
        self.setup_training_tab()
        self.setup_evaluation_tab()
        self.setup_metrics_tab()
        self.setup_dataset_tab()
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        if not TRANSLATION_AVAILABLE:
            self.status_bar.config(text="⚠️ Translation requires additional libraries")
        elif not TRAINING_AVAILABLE:
            self.status_bar.config(text="⚠️ Training features require additional libraries")
        elif self.dataset:
            self.status_bar.config(text=f"Loaded {len(self.dataset)} translation pairs from dataset")

    def setup_translation_tab(self):
        main_frame = tk.Frame(self.translation_tab, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            main_frame,
            text="English to Arabic Translation System",
            font=('Arial', 16, 'bold')
        ).pack(pady=20)
        model_frame = tk.LabelFrame(main_frame, text="Translation Model", padx=10, pady=10)
        model_frame.pack(fill=tk.X, pady=10)
        self.model_var = tk.StringVar(value="pretrained")
        tk.Radiobutton(
            model_frame,
            text="Use Pre-trained Model",
            variable=self.model_var,
            value="pretrained"
        ).pack(anchor='w')
        tk.Radiobutton(
            model_frame,
            text="Use Fine-tuned Model",
            variable=self.model_var,
            value="finetuned"
        ).pack(anchor='w')
        translation_frame = tk.LabelFrame(main_frame, text="English to Arabic Translation", padx=10, pady=10)
        translation_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        tk.Label(translation_frame, text="Enter English Text:").pack(anchor='w', pady=(10, 5))
        self.text_input = tk.Text(translation_frame, height=6, width=60, font=('Arial', 12))
        self.text_input.pack(fill=tk.X, padx=5, pady=5)
        preproc_frame = tk.Frame(translation_frame)
        preproc_frame.pack(fill=tk.X, anchor='w', padx=5)
        self.clean_text_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            preproc_frame,
            text="Apply text preprocessing",
            variable=self.clean_text_var
        ).pack(side=tk.LEFT)
        button_frame = tk.Frame(translation_frame)
        button_frame.pack(fill=tk.X, pady=10)
        self.translate_button = tk.Button(
            button_frame,
            text="Translate to Arabic",
            command=self.translate_input_en_ar,
            bg="#4CAF50",
            fg="white",
            padx=10,
            pady=5,
            font=('Arial', 10, 'bold')
        )
        self.translate_button.pack(side=tk.LEFT, padx=5)

        self.save_trans_button = tk.Button(
            button_frame,
            text="Save Translation",
            command=self.save_translation,
            bg="#2196F3",
            fg="white",
            padx=10,
            pady=5
        )
        self.save_trans_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_fields,
            bg="#f44336",
            fg="white",
            padx=10,
            pady=5
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.load_dataset_button = tk.Button(
            button_frame,
            text="Load Dataset",
            command=self.browse_dataset,
            bg="#FF9800",
            fg="white",
            padx=10,
            pady=5
        )
        self.load_dataset_button.pack(side=tk.LEFT, padx=5)

        if not TRANSLATION_AVAILABLE:
            self.install_button = tk.Button(
                button_frame,
                text="Install Dependencies",
                command=self.show_install_instructions,
                bg="#2196F3",
                fg="white",
                padx=10,
                pady=5
            )
            self.install_button.pack(side=tk.LEFT, padx=5)

        if self.dataset:
            self.sample_button = tk.Button(
                button_frame,
                text="Use Sample",
                command=self.use_dataset_sample,
                bg="#9C27B0",
                fg="white",
                padx=10,
                pady=5
            )
            self.sample_button.pack(side=tk.LEFT, padx=5)

        tk.Label(translation_frame, text="Arabic Translation:").pack(anchor='w', pady=(10, 5))
        self.translation_output = tk.Text(
            translation_frame,
            height=8,
            wrap=tk.WORD,
            font=('Arial', 12),
            bg="#f0f0f0"
        )
        self.translation_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_preprocessing_tab(self):
        main_frame = tk.Frame(self.preprocessing_tab, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            main_frame,
            text="Text Preprocessing",
            font=('Arial', 16, 'bold')
        ).pack(pady=10)
        input_frame = tk.LabelFrame(main_frame, text="Input Text", padx=10, pady=10)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        lang_frame = tk.Frame(input_frame)
        lang_frame.pack(fill=tk.X, pady=5)
        tk.Label(lang_frame, text="Language:").pack(side=tk.LEFT, padx=5)
        self.preproc_lang_var = tk.StringVar(value="english")
        tk.Radiobutton(
            lang_frame,
            text="English",
            variable=self.preproc_lang_var,
            value="english"
        ).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(
            lang_frame,
            text="Arabic",
            variable=self.preproc_lang_var,
            value="arabic"
        ).pack(side=tk.LEFT, padx=5)
        tk.Label(input_frame, text="Enter Text to Process:").pack(anchor='w', pady=(10, 5))
        self.preproc_input = tk.Text(input_frame, height=6, width=60, font=('Arial', 12))
        self.preproc_input.pack(fill=tk.X, padx=5, pady=5)
        options_frame = tk.LabelFrame(main_frame, text="Preprocessing Options", padx=10, pady=10)
        options_frame.pack(fill=tk.X, pady=10)
        self.en_options_frame = tk.Frame(options_frame)
        self.en_options_frame.pack(fill=tk.X, pady=5)
        self.en_lowercase_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.en_options_frame,
            text="Convert to lowercase",
            variable=self.en_lowercase_var
        ).pack(anchor='w')
        self.en_remove_special_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.en_options_frame,
            text="Remove special characters",
            variable=self.en_remove_special_var
        ).pack(anchor='w')
        self.en_remove_urls_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.en_options_frame,
            text="Remove URLs",
            variable=self.en_remove_urls_var
        ).pack(anchor='w')
        self.ar_options_frame = tk.Frame(options_frame)
        self.ar_remove_diacritics_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.ar_options_frame,
            text="Remove diacritics (tashkeel)",
            variable=self.ar_remove_diacritics_var
        ).pack(anchor='w')
        self.ar_remove_tatweel_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.ar_options_frame,
            text="Remove tatweel (stretching)",
            variable=self.ar_remove_tatweel_var
        ).pack(anchor='w')
        self.ar_normalize_chars_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.ar_options_frame,
            text="Normalize Arabic characters",
            variable=self.ar_normalize_chars_var
        ).pack(anchor='w')
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        self.process_button = tk.Button(
            button_frame,
            text="Apply Preprocessing",
            command=self.apply_preprocessing,
            bg="#4CAF50",
            fg="white",
            padx=10,
            pady=5
        )
        self.process_button.pack(side=tk.LEFT, padx=5)
        self.clear_preproc_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_preprocessing_fields,
            bg="#f44336",
            fg="white",
            padx=10,
            pady=5
        )
        self.clear_preproc_button.pack(side=tk.LEFT, padx=5)
        output_frame = tk.LabelFrame(main_frame, text="Processed Text", padx=10, pady=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.preproc_output = tk.Text(
            output_frame,
            height=8,
            wrap=tk.WORD,
            font=('Arial', 12),
            bg="#f0f0f0"
        )
        self.preproc_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_training_tab(self):
        main_frame = tk.Frame(self.training_tab, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            main_frame,
            text="Model Training",
            font=('Arial', 16, 'bold')
        ).pack(pady=10)
        if not TRAINING_AVAILABLE:
            tk.Label(
                main_frame,
                text="Training features require additional libraries:\n"
                     "pip install transformers torch sentencepiece datasets",
                fg="red",
                font=('Arial', 12)
            ).pack(pady=20)
            return

        settings_frame = tk.LabelFrame(main_frame, text="Training Settings", padx=10, pady=10)
        settings_frame.pack(fill=tk.X, pady=10)
        tk.Label(settings_frame, text="Base Model:").grid(row=0, column=0, sticky='w', pady=5)
        self.base_model_entry = tk.Entry(settings_frame, width=50)
        self.base_model_entry.grid(row=0, column=1, sticky='we', pady=5, padx=5)
        self.base_model_entry.insert(0, DEFAULT_MODEL_PATH)
        tk.Label(settings_frame, text="Output Directory:").grid(row=1, column=0, sticky='w', pady=5)
        self.output_dir_entry = tk.Entry(settings_frame, width=50)
        self.output_dir_entry.grid(row=1, column=1, sticky='we', pady=5, padx=5)
        self.output_dir_entry.insert(0, DEFAULT_SAVED_MODEL_PATH)
        tk.Button(
            settings_frame,
            text="Browse",
            command=self.browse_output_dir,
            width=10
        ).grid(row=1, column=2, padx=5)
        tk.Label(settings_frame, text="Epochs:").grid(row=2, column=0, sticky='w', pady=5)
        self.epochs_entry = tk.Entry(settings_frame, width=10)
        self.epochs_entry.grid(row=2, column=1, sticky='w', pady=5, padx=5)
        self.epochs_entry.insert(0, str(EPOCHS))
        tk.Label(settings_frame, text="Batch Size:").grid(row=3, column=0, sticky='w', pady=5)
        self.batch_size_entry = tk.Entry(settings_frame, width=10)
        self.batch_size_entry.grid(row=3, column=1, sticky='w', pady=5, padx=5)
        self.batch_size_entry.insert(0, str(BATCH_SIZE))
        tk.Label(settings_frame, text="Learning Rate:").grid(row=4, column=0, sticky='w', pady=5)
        self.learning_rate_entry = tk.Entry(settings_frame, width=10)
        self.learning_rate_entry.grid(row=4, column=1, sticky='w', pady=5, padx=5)
        self.learning_rate_entry.insert(0, "5e-5")
        dataset_frame = tk.LabelFrame(main_frame, text="Dataset Information", padx=10, pady=10)
        dataset_frame.pack(fill=tk.X, pady=10)
        self.dataset_info_label = tk.Label(
            dataset_frame,
            text=f"Loaded {len(self.dataset)} translation pairs" if self.dataset else "No dataset loaded",
            wraplength=600,
            justify=tk.LEFT
        )
        self.dataset_info_label.pack(anchor='w')
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        self.train_button = tk.Button(
            button_frame,
            text="Start Training",
            command=self.start_training,
            bg="#4CAF50",
            fg="white",
            padx=10,
            pady=5,
            font=('Arial', 10, 'bold')
        )
        self.train_button.pack(side=tk.LEFT, padx=5)
        log_frame = tk.LabelFrame(main_frame, text="Training Log", padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.training_log = tk.Text(
            log_frame,
            height=10,
            wrap=tk.WORD,
            font=('Courier', 10),
            bg="#f0f0f0"
        )
        self.training_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar = tk.Scrollbar(log_frame, command=self.training_log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.training_log.config(yscrollcommand=scrollbar.set)

    def setup_evaluation_tab(self):
        main_frame = tk.Frame(self.evaluation_tab, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            main_frame,
            text="Translation Evaluation",
            font=('Arial', 16, 'bold')
        ).pack(pady=10)
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        self.eval_button = tk.Button(
            control_frame,
            text="Run Evaluation",
            command=self.run_evaluation,
            bg="#4CAF50",
            fg="white",
            padx=10,
            pady=5,
            font=('Arial', 10, 'bold')
        )
        self.eval_button.pack(side=tk.LEFT, padx=5)
        self.save_eval_button = tk.Button(
            control_frame,
            text="Save Results",
            command=self.save_evaluation_results,
            bg="#2196F3",
            fg="white",
            padx=10,
            pady=5
        )
        self.save_eval_button.pack(side=tk.LEFT, padx=5)
        results_frame = tk.LabelFrame(main_frame, text="Evaluation Results", padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        metrics_frame = tk.Frame(results_frame)
        metrics_frame.pack(fill=tk.X, pady=10)
        left_metrics = tk.Frame(metrics_frame)
        left_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_metrics = tk.Frame(metrics_frame)
        right_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(left_metrics, text="Avg. Source Length:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=2)
        self.avg_source_length_var = tk.StringVar(value="N/A")
        tk.Label(left_metrics, textvariable=self.avg_source_length_var).grid(row=0, column=1, sticky='w', pady=2)
        tk.Label(left_metrics, text="Avg. Target Length:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky='w', pady=2)
        self.avg_target_length_var = tk.StringVar(value="N/A")
        tk.Label(left_metrics, textvariable=self.avg_target_length_var).grid(row=1, column=1, sticky='w', pady=2)
        tk.Label(left_metrics, text="Mean Squared Error:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=2)
        self.mse_var = tk.StringVar(value="N/A")
        tk.Label(left_metrics, textvariable=self.mse_var).grid(row=2, column=1, sticky='w', pady=2)
        tk.Label(right_metrics, text="Character Ratio:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=2)
        self.char_ratio_var = tk.StringVar(value="N/A")
        tk.Label(right_metrics, textvariable=self.char_ratio_var).grid(row=0, column=1, sticky='w', pady=2)
        tk.Label(right_metrics, text="Translation Count:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky='w', pady=2)
        self.trans_count_var = tk.StringVar(value="0")
        tk.Label(right_metrics, textvariable=self.trans_count_var).grid(row=1, column=1, sticky='w', pady=2)
        tk.Label(right_metrics, text="Mean Absolute Error:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=2)
        self.mae_var = tk.StringVar(value="N/A")
        tk.Label(right_metrics, textvariable=self.mae_var).grid(row=2, column=1, sticky='w', pady=2)
        tk.Label(right_metrics, text="Accuracy:", font=('Arial', 10, 'bold')).grid(row=3, column=0, sticky='w', pady=2)
        self.accuracy_var = tk.StringVar(value="N/A")
        tk.Label(right_metrics, textvariable=self.accuracy_var).grid(row=3, column=1, sticky='w', pady=2)
        if BLEU_AVAILABLE:
            tk.Label(right_metrics, text="BLEU Score:", font=('Arial', 10, 'bold')).grid(row=4, column=0, sticky='w', pady=2)
            self.bleu_var = tk.StringVar(value="N/A")
            tk.Label(right_metrics, textvariable=self.bleu_var).grid(row=4, column=1, sticky='w', pady=2)

        examples_frame = tk.LabelFrame(results_frame, text="Translation Examples", padx=10, pady=10)
        examples_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.examples_tree = ttk.Treeview(examples_frame, columns=('English', 'Arabic'), show='headings')
        self.examples_tree.heading('English', text='English Text')
        self.examples_tree.heading('Arabic', text='Arabic Translation')
        self.examples_tree.column('English', width=200)
        self.examples_tree.column('Arabic', width=200)
        self.examples_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        x_scrollbar = ttk.Scrollbar(examples_frame, orient=tk.HORIZONTAL, command=self.examples_tree.xview)
        self.examples_tree.configure(xscrollcommand=x_scrollbar.set)
        x_scrollbar.pack(fill=tk.X)

    def setup_metrics_tab(self):
        main_frame = tk.Frame(self.metrics_tab, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            main_frame,
            text="Translation Metrics",
            font=('Arial', 16, 'bold')
        ).pack(pady=10)
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        self.metrics_button = tk.Button(
            control_frame,
            text="Generate Metrics",
            command=self.calculate_and_display_metrics,
            bg="#4CAF50",
            fg="white",
            padx=10,
            pady=5,
            font=('Arial', 10, 'bold')
        )
        self.metrics_button.pack(side=tk.LEFT, padx=5)
        self.metrics_info = ttk.Label(main_frame, font=("Arial", 12))
        self.metrics_info.pack(pady=10)
        self.metrics_frame = tk.Frame(main_frame)
        self.metrics_frame.pack(fill=tk.BOTH, expand=True, pady=10)

    def setup_dataset_tab(self):
        main_frame = tk.Frame(self.dataset_tab, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            main_frame,
            text="Dataset Viewer",
            font=('Arial', 16, 'bold')
        ).pack(pady=10)
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        self.load_dataset_btn = tk.Button(
            control_frame,
            text="Load Dataset",
            command=self.browse_dataset,
            bg="#4CAF50",
            fg="white",
            padx=10,
            pady=5
        )
        self.load_dataset_btn.pack(side=tk.LEFT, padx=5)
        self.dataset_info = ttk.Label(main_frame, font=("Arial", 12))
        if self.dataset:
            self.dataset_info.config(text=f"Loaded {len(self.dataset)} translation pairs from:\n{self.dataset_path}")
        else:
            self.dataset_info.config(text="No dataset loaded.")
        self.dataset_info.pack(pady=10)
        viewer_frame = tk.LabelFrame(main_frame, text="Dataset Entries", padx=10, pady=10)
        viewer_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.dataset_tree = ttk.Treeview(viewer_frame, columns=('ID', 'English', 'Arabic'), show='headings')
        self.dataset_tree.heading('ID', text='ID')
        self.dataset_tree.heading('English', text='English Text')
        self.dataset_tree.heading('Arabic', text='Arabic Translation')
        self.dataset_tree.column('ID', width=50)
        self.dataset_tree.column('English', width=200)
        self.dataset_tree.column('Arabic', width=200)
        y_scrollbar = ttk.Scrollbar(viewer_frame, orient=tk.VERTICAL, command=self.dataset_tree.yview)
        self.dataset_tree.configure(yscrollcommand=y_scrollbar.set)
        x_scrollbar = ttk.Scrollbar(viewer_frame, orient=tk.HORIZONTAL, command=self.dataset_tree.xview)
        self.dataset_tree.configure(xscrollcommand=x_scrollbar.set)
        self.dataset_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.populate_dataset_tree()

    def populate_dataset_tree(self):
        for item in self.dataset_tree.get_children():
            self.dataset_tree.delete(item)

        if self.dataset:
            for i, (en_text, ar_text) in enumerate(self.dataset):
                self.dataset_tree.insert('', 'end', values=(i + 1, en_text, ar_text))

    def translate_input_en_ar(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter text to translate")
            return

        self.status_bar.config(text="Translating...")
        self.translation_output.delete(1.0, tk.END)
        self.translation_output.insert(tk.END, "Translating...")
        self.root.update()

        try:
            if self.model_var.get() == "pretrained":
                model_path = DEFAULT_MODEL_PATH
            else:
                model_path = self.saved_model_path

            translated_text = translate_text_en_to_ar(text, model_path)
            self.translation_output.delete(1.0, tk.END)
            self.translation_output.insert(tk.END, translated_text)
            self.status_bar.config(text="Translation complete")
            self.translations.append((text, translated_text))
        except Exception as e:
            self.translation_output.delete(1.0, tk.END)
            self.translation_output.insert(tk.END, f"Translation error: {str(e)}")
            self.status_bar.config(text="Translation failed")

    def save_translation(self):
        source_text = self.text_input.get("1.0", tk.END).strip()
        translated_text = self.translation_output.get("1.0", tk.END).strip()
        if not source_text or not translated_text:
            messagebox.showwarning("Warning", "No translation to save")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Translation"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("Source (English):\n")
                f.write(f"{source_text}\n\n")
                f.write("Translation (Arabic):\n")
                f.write(f"{translated_text}\n")
            messagebox.showinfo("Success", f"Translation saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save translation: {str(e)}")

    def clear_fields(self):
        self.text_input.delete("1.0", tk.END)
        self.translation_output.delete("1.0", tk.END)

    def show_install_instructions(self):
        messagebox.showinfo(
            "Installation Instructions",
            "To enable translation features, please install the following packages:\n\n"
            "pip install transformers sentencepiece\n\n"
            "For training features, also install:\n"
            "pip install torch datasets\n\n"
            "For Arabic text processing:\n"
            "pip install pyarabic\n\n"
            "For BLEU score calculation:\n"
            "pip install sacrebleu\n\n"
            "After installation, restart the application."
        )

    def browse_dataset(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Select Dataset File"
        )
        if file_path:
            self.dataset_path = file_path
            self.dataset, message = load_and_preprocess_dataset(file_path)
            if self.dataset:
                self.dataset_info.config(
                    text=f"Loaded {len(self.dataset)} translation pairs from:\n{self.dataset_path}")
                messagebox.showinfo("Success", f"Loaded {len(self.dataset)} translation pairs from dataset")
                self.populate_dataset_tree()
                self.status_bar.config(text=f"Loaded {len(self.dataset)} translation pairs from dataset")
                if hasattr(self, 'dataset_info_label'):
                    self.dataset_info_label.config(text=f"Loaded {len(self.dataset)} translation pairs")
                found_sample_button = False
                for widget in self.translation_tab.winfo_descendants():
                    if isinstance(widget, tk.Button) and widget.cget('text') == "Use Sample":
                        found_sample_button = True
                        break

                if not found_sample_button:
                    for widget in self.translation_tab.winfo_descendants():
                        if isinstance(widget, tk.Frame):
                            for child in widget.winfo_children():
                                if isinstance(child, tk.Button) and child.cget('text') == "Load Dataset":
                                    self.sample_button = tk.Button(
                                        widget,
                                        text="Use Sample",
                                        command=self.use_dataset_sample,
                                        bg="#9C27B0",
                                        fg="white",
                                        padx=10,
                                        pady=5
                                    )
                                    self.sample_button.pack(side=tk.LEFT, padx=5)
                                    break
            else:
                messagebox.showwarning("Warning", message)

    def use_dataset_sample(self):
        if not self.dataset:
            messagebox.showwarning("Warning", "No dataset loaded")
            return
        sample = random.choice(self.dataset)
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", sample[0])
        self.translation_output.delete("1.0", tk.END)

    def apply_preprocessing(self):
        text = self.preproc_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter text to process")
            return

        lang = self.preproc_lang_var.get()
        try:
            if lang == "english":
                if self.en_lowercase_var.get():
                    text = text.lower()

                if self.en_remove_special_var.get():
                    text = re.sub(r'[^a-zA-Z0-9\s\.,;:!?\'"-]', '', text)

                if self.en_remove_urls_var.get():
                    text = re.sub(r'http\S+|www\S+', '', text)

                text = re.sub(r'\s+', ' ', text).strip()
            else:
                if ARABIC_PREPROCESSING:
                    if self.ar_remove_diacritics_var.get():
                        text = araby.strip_tashkeel(text)

                    if self.ar_remove_tatweel_var.get():
                        text = araby.strip_tatweel(text)

                    if self.ar_normalize_chars_var.get():
                        text = araby.normalize_hamza(text)
                        text = araby.normalize_lamalef(text)
                else:
                    if self.ar_remove_diacritics_var.get():
                        text = re.sub(r'[\u064B-\u065F]', '', text)  # Basic diacritics removal

                    if self.ar_remove_tatweel_var.get():
                        text = re.sub(r'ـ', '', text)

                    if self.ar_normalize_chars_var.get():
                        text = re.sub(r'[أإآ]', 'ا', text)

                text = re.sub(r'\s+', ' ', text).strip()

            self.preproc_output.delete("1.0", tk.END)
            self.preproc_output.insert("1.0", text)
        except Exception as e:
            messagebox.showerror("Error", f"Error during preprocessing: {str(e)}")

    def clear_preprocessing_fields(self):
        self.preproc_input.delete("1.0", tk.END)
        self.preproc_output.delete("1.0", tk.END)

    def browse_output_dir(self):
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, dir_path)

    def start_training(self):
        if not self.dataset or len(self.dataset) < 10:
            messagebox.showwarning("Warning", "Insufficient data for training. Please load a larger dataset.")
            return

        try:
            model_path = self.base_model_entry.get().strip()
            output_dir = self.output_dir_entry.get().strip()
            epochs = int(self.epochs_entry.get())
            batch_size = int(self.batch_size_entry.get())
            learning_rate = float(self.learning_rate_entry.get())
            if not model_path:
                messagebox.showwarning("Warning", "Please specify a base model")
                return

            if not output_dir:
                messagebox.showwarning("Warning", "Please specify an output directory")
                return

            os.makedirs(output_dir, exist_ok=True)
            self.saved_model_path = output_dir
            self.training_log.delete("1.0", tk.END)
            self.training_log.insert(tk.END, "Starting training...\n")
            def training_thread():
                def log_callback(message):
                    self.training_log.insert(tk.END, message + "\n")
                    self.training_log.see(tk.END)
                    self.root.update()

                success, message = fine_tune_model(
                    self.dataset,
                    model_path=model_path,
                    output_dir=output_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    progress_callback=log_callback
                )

                if success:
                    log_callback(f"Training completed successfully!\n{message}")
                    messagebox.showinfo("Success", "Model training completed successfully!")
                else:
                    log_callback(f"Training failed:\n{message}")
                    messagebox.showerror("Error", f"Training failed:\n{message}")

            threading.Thread(target=training_thread, daemon=True).start()

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error starting training: {str(e)}")

    def run_evaluation(self):
        if not self.translations:
            messagebox.showwarning("Warning", "No translations to evaluate. Please translate some text first.")
            return

        self.show_loading("Evaluating translations...")
        try:
            self.evaluation_results = evaluate_translation(self.translations)
            self.avg_source_length_var.set(f"{self.evaluation_results['avg_source_length']:.2f}")
            self.avg_target_length_var.set(f"{self.evaluation_results['avg_target_length']:.2f}")
            self.char_ratio_var.set(f"{self.evaluation_results['char_ratio']:.2f}")
            self.trans_count_var.set(f"{len(self.translations)}")
            self.mse_var.set(f"{self.evaluation_results['mse']:.4f}")
            self.mae_var.set(f"{self.evaluation_results['mae']:.4f}")
            self.accuracy_var.set(f"{self.evaluation_results['accuracy']:.2%}")
            if 'bleu' in self.evaluation_results:
                if hasattr(self, 'bleu_var'):
                    self.bleu_var.set(f"{self.evaluation_results['bleu']:.2f}")

            self.examples_tree.delete(*self.examples_tree.get_children())
            for source, target in self.evaluation_results['examples']:
                self.examples_tree.insert('', 'end', values=(source, target))

            self.evaluation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.hide_loading()
            messagebox.showinfo("Success", "Translation evaluation completed successfully!")
            self.notebook.select(self.evaluation_tab)
        except Exception as e:
            self.hide_loading()
            error_message = f"An error occurred during evaluation: {str(e)}"
            messagebox.showerror("Evaluation Error", error_message)
            print(f"Evaluation error: {traceback.format_exc()}")

    def calculate_and_display_metrics(self):
        if not self.translations:
            messagebox.showwarning("Warning", "No translations to analyze. Please translate some text first.")
            return

        self.show_loading("Calculating metrics...")
        try:
            total_translations = len(self.translations)
            source_lengths = [len(pair[0]) for pair in self.translations]
            target_lengths = [len(pair[1]) for pair in self.translations]
            mse_values = []
            mae_values = []
            accuracy_values = []
            bleu_scores = []
            for source, target in self.translations:
                mse = calculate_mse(source, target)
                mae = calculate_mae(source, target)
                accuracy = calculate_accuracy(source, target)
                mse_values.append(mse)
                mae_values.append(mae)
                accuracy_values.append(accuracy)
                if BLEU_AVAILABLE:
                    bleu = calculate_bleu_score(source, target)
                    if bleu is not None:
                        bleu_scores.append(bleu)

            metrics_text = (
                f"Total Translations: {total_translations}\n"
                f"Average Source Length: {np.mean(source_lengths):.2f} characters\n"
                f"Average Target Length: {np.mean(target_lengths):.2f} characters\n"
                f"Average Character Ratio: {np.mean(target_lengths) / np.mean(source_lengths):.2f}\n"
                f"Mean Squared Error (MSE): {np.mean(mse_values):.4f}\n"
                f"Mean Absolute Error (MAE): {np.mean(mae_values):.4f}\n"
                f"Average Accuracy: {np.mean(accuracy_values):.2%}"
            )
            if BLEU_AVAILABLE and bleu_scores:
                metrics_text += f"\nAverage BLEU Score: {np.mean(bleu_scores):.2f}"

            self.metrics_info.config(text=metrics_text)
            for widget in self.metrics_frame.winfo_children():
                widget.destroy()

            fig = Figure(figsize=(12, 10), dpi=100)
            ax1 = fig.add_subplot(3, 2, 1)
            ax2 = fig.add_subplot(3, 2, 2)
            ax3 = fig.add_subplot(3, 2, 3)
            ax4 = fig.add_subplot(3, 2, 4)
            ax5 = fig.add_subplot(3, 2, 5)
            ax6 = fig.add_subplot(3, 2, 6)
            languages = ['English', 'Arabic']
            avg_lengths = [np.mean(source_lengths), np.mean(target_lengths)]
            colors = ['blue', 'green']
            ax1.bar(languages, avg_lengths, color=colors)
            ax1.set_title('Average Text Length')
            ax1.set_ylabel('Characters')
            ax2.hist([source_lengths, target_lengths], bins=20, color=['blue', 'green'],
                     label=['English', 'Arabic'], alpha=0.7)
            ax2.set_title('Length Distribution')
            ax2.set_xlabel('Length')
            ax2.set_ylabel('Count')
            ax2.legend()
            ax3.plot(mse_values, color='red', marker='o', linestyle='-', markersize=3)
            ax3.set_title('Mean Squared Error (MSE)')
            ax3.set_xlabel('Translation Index')
            ax3.set_ylabel('MSE Value')
            ax4.plot(mae_values, color='blue', marker='o', linestyle='-', markersize=3)
            ax4.set_title('Mean Absolute Error (MAE)')
            ax4.set_xlabel('Translation Index')
            ax4.set_ylabel('MAE Value')
            ax5.plot(accuracy_values, color='green', marker='o', linestyle='-', markersize=3)
            ax5.set_title('Accuracy per Translation')
            ax5.set_xlabel('Translation Index')
            ax5.set_ylabel('Accuracy')
            ax6.hist(accuracy_values, bins=20, color='purple')
            ax6.set_title('Accuracy Distribution')
            ax6.set_xlabel('Accuracy')
            ax6.set_ylabel('Count')
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.metrics_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            toolbar = NavigationToolbar2Tk(canvas, self.metrics_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.hide_loading()
            self.notebook.select(self.metrics_tab)

        except Exception as e:
            self.hide_loading()
            error_message = f"An error occurred calculating metrics: {str(e)}"
            messagebox.showerror("Metrics Error", error_message)
            print(f"Metrics error: {traceback.format_exc()}")

    def save_evaluation_results(self):
        if self.evaluation_results is None:
            messagebox.showwarning("Warning", "No evaluation results to save. Please run evaluation first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Evaluation Results"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Translation Evaluation Results - {self.evaluation_timestamp}\n\n")
                f.write(f"Number of Translations: {len(self.translations)}\n")
                f.write(f"Average Source Length: {self.evaluation_results['avg_source_length']:.2f} characters\n")
                f.write(f"Average Target Length: {self.evaluation_results['avg_target_length']:.2f} characters\n")
                f.write(f"Character Ratio (Arabic/English): {self.evaluation_results['char_ratio']:.2f}\n")
                f.write(f"Mean Squared Error (MSE): {self.evaluation_results['mse']:.4f}\n")
                f.write(f"Mean Absolute Error (MAE): {self.evaluation_results['mae']:.4f}\n")
                f.write(f"Accuracy: {self.evaluation_results['accuracy']:.2%}\n")

                if 'bleu' in self.evaluation_results:
                    f.write(f"BLEU Score: {self.evaluation_results['bleu']:.2f}\n")

                f.write("\nTranslation Examples:\n")
                for i, (source, target) in enumerate(self.evaluation_results['examples']):
                    f.write(f"Example {i + 1}:\n")
                    f.write(f"  English: {source}\n")
                    f.write(f"  Arabic: {target}\n\n")

            messagebox.showinfo("Success", f"Evaluation results saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def show_loading(self, message="Loading..."):
        if hasattr(self, 'loading_window') and self.loading_window is not None:
            self.loading_window.destroy()

        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.title("Processing")
        self.loading_window.geometry("300x100")
        self.loading_window.resizable(False, False)
        self.loading_window.transient(self.root)
        self.loading_window.grab_set()
        self.loading_window.update_idletasks()
        width = self.loading_window.winfo_width()
        height = self.loading_window.winfo_height()
        x = (self.loading_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.loading_window.winfo_screenheight() // 2) - (height // 2)
        self.loading_window.geometry(f'+{x}+{y}')
        message_label = ttk.Label(self.loading_window, text=message)
        message_label.pack(pady=10)
        progress = ttk.Progressbar(self.loading_window, mode='indeterminate')
        progress.pack(fill='x', padx=20, pady=10)
        progress.start(10)
        self.loading_window.update()

    def hide_loading(self):
        if hasattr(self, 'loading_window') and self.loading_window is not None:
            self.loading_window.destroy()
            self.loading_window = None

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedTranslationApp(root)
    root.mainloop()