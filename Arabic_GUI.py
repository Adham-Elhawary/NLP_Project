import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime
import traceback
import random

try:
    from transformers import MarianMTModel, MarianTokenizer

    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

BATCH_SIZE = 32
EPOCHS = 10
TEST_SPLIT = 0.2
DEFAULT_DATASET_PATH = r"C:\Users\Adham\Downloads\nlp_2\ara_eng.txt"


def load_dataset(file_path=None):
    try:
        if not file_path or not os.path.exists(file_path):
            return None, f"ملف مجموعة البيانات غير موجود: {file_path}"

        data_pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                en_text = lines[i].strip()
                ar_text = lines[i + 1].strip()
                if en_text and ar_text:
                    data_pairs.append((en_text, ar_text))

        return data_pairs, f"تم تحميل {len(data_pairs)} زوج ترجمة بنجاح"
    except Exception as e:
        return None, f"خطأ في تحميل مجموعة البيانات: {str(e)}"

def translate_text_en_to_ar(text):
    try:
        print(f"جاري الترجمة: '{text}'")

        if not TRANSLATION_AVAILABLE:
            return "مكتبات الترجمة غير متوفرة. يرجى تثبيت الحزم المطلوبة."

        model_name = 'Helsinki-NLP/opus-mt-en-ar'
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated = model.generate(**inputs)
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
            print(f"نتيجة الترجمة: '{result}'")
        except Exception as e:
            print(f"خطأ في استخدام MarianMT: {str(e)}")
            result = f"غير قادر على الترجمة إلى العربية. خطأ: {str(e)}\n\nلإصلاح هذه المشكلة، يرجى تثبيت المكتبات المطلوبة:\npip install transformers sentencepiece"

        return result
    except Exception as e:
        print(f"خطأ في الترجمة: {str(e)}")
        return f"خطأ في الترجمة: {str(e)}"


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
    for source, target in translations:
        mse = calculate_mse(source, target)
        mae = calculate_mae(source, target)
        mse_values.append(mse)
        mae_values.append(mae)

    evaluation_results = {
        'avg_source_length': avg_source_length,
        'avg_target_length': avg_target_length,
        'char_ratio': char_ratio,
        'mse': np.mean(mse_values) if mse_values else 0,
        'mae': np.mean(mae_values) if mae_values else 0,
        'examples': translations[:5]
    }
    return evaluation_results

class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("نظام الترجمة الإنجليزية-العربية")
        self.root.geometry("1000x800")
        self.loading_window = None
        self.evaluation_results = None
        self.evaluation_timestamp = None
        self.translations = []
        self.dataset = []
        self.dataset_path = DEFAULT_DATASET_PATH
        global TRANSLATION_AVAILABLE
        try:
            import transformers
            import sentencepiece
            TRANSLATION_AVAILABLE = True
        except ImportError:
            TRANSLATION_AVAILABLE = False
            messagebox.showwarning(
                "الترجمة غير متوفرة",
                "بعض المكتبات المطلوبة للترجمة مفقودة. "
                "ستوفر ميزة الترجمة تعليمات التثبيت بدلاً من ذلك.\n\n"
                "قم بتثبيت المكتبات المطلوبة باستخدام:\n"
                "pip install transformers sentencepiece"
            )
        self.load_dataset()
        self.create_widgets()

    def load_dataset(self):
        if os.path.exists(self.dataset_path):
            self.dataset, message = load_dataset(self.dataset_path)
            if not self.dataset:
                messagebox.showwarning("تحذير مجموعة البيانات", message)
        else:
            messagebox.showwarning(
                "مجموعة البيانات غير موجودة",
                f"لم يتم العثور على ملف مجموعة البيانات في: {self.dataset_path}\n"
                "لا يزال بإمكانك استخدام التطبيق، ولكن قد تكون بعض الميزات محدودة."
            )
    def create_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.translation_tab = ttk.Frame(self.notebook)
        self.evaluation_tab = ttk.Frame(self.notebook)
        self.metrics_tab = ttk.Frame(self.notebook)
        self.dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.translation_tab, text="الترجمة")
        self.notebook.add(self.evaluation_tab, text="التقييم")
        self.notebook.add(self.metrics_tab, text="المقاييس")
        self.notebook.add(self.dataset_tab, text="مجموعة البيانات")
        self.setup_translation_tab()
        self.setup_evaluation_tab()
        self.setup_metrics_tab()
        self.setup_dataset_tab()
        self.status_bar = tk.Label(
            self.root,
            text="جاهز",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        if not TRANSLATION_AVAILABLE:
            self.status_bar.config(text="⚠️ الترجمة تتطلب مكتبات إضافية")
        if self.dataset:
            self.status_bar.config(text=f"تم تحميل {len(self.dataset)} زوج ترجمة من مجموعة البيانات")

    def setup_translation_tab(self):
        main_frame = tk.Frame(self.translation_tab, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            main_frame,
            text="نظام الترجمة من الإنجليزية إلى العربية",
            font=('Arial', 16, 'bold')
        ).pack(pady=20)
        translation_frame = tk.LabelFrame(main_frame, text="الترجمة من الإنجليزية إلى العربية", padx=10, pady=10)
        translation_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        tk.Label(translation_frame, text="أدخل النص الإنجليزي:").pack(anchor='w', pady=(10, 5))
        self.text_input = tk.Text(translation_frame, height=6, width=60, font=('Arial', 12))
        self.text_input.pack(fill=tk.X, padx=5, pady=5)
        button_frame = tk.Frame(translation_frame)
        button_frame.pack(fill=tk.X, pady=10)
        self.translate_button = tk.Button(
            button_frame,
            text="ترجم إلى العربية",
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
            text="حفظ الترجمة",
            command=self.save_translation,
            bg="#2196F3",
            fg="white",
            padx=10,
            pady=5
        )
        self.save_trans_button.pack(side=tk.LEFT, padx=5)
        self.clear_button = tk.Button(
            button_frame,
            text="مسح",
            command=self.clear_fields,
            bg="#f44336",
            fg="white",
            padx=10,
            pady=5
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.load_dataset_button = tk.Button(
            button_frame,
            text="تحميل مجموعة البيانات",
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
                text="تثبيت التبعيات",
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
                text="استخدم عينة",
                command=self.use_dataset_sample,
                bg="#9C27B0",
                fg="white",
                padx=10,
                pady=5
            )
            self.sample_button.pack(side=tk.LEFT, padx=5)
        tk.Label(translation_frame, text="الترجمة العربية:").pack(anchor='w', pady=(10, 5))
        self.translation_output = tk.Text(
            translation_frame,
            height=8,
            wrap=tk.WORD,
            font=('Arial', 12),
            bg="#f0f0f0"
        )
        self.translation_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_evaluation_tab(self):
        main_frame = tk.Frame(self.evaluation_tab, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            main_frame,
            text="تقييم الترجمة",
            font=('Arial', 16, 'bold')
        ).pack(pady=10)
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        self.eval_button = tk.Button(
            control_frame,
            text="تشغيل التقييم",
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
            text="حفظ النتائج",
            command=self.save_evaluation_results,
            bg="#2196F3",
            fg="white",
            padx=10,
            pady=5
        )
        self.save_eval_button.pack(side=tk.LEFT, padx=5)
        results_frame = tk.LabelFrame(main_frame, text="نتائج التقييم", padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        metrics_frame = tk.Frame(results_frame)
        metrics_frame.pack(fill=tk.X, pady=10)
        left_metrics = tk.Frame(metrics_frame)
        left_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_metrics = tk.Frame(metrics_frame)
        right_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(left_metrics, text="متوسط طول النص الأصلي:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=2)
        self.avg_source_length_var = tk.StringVar(value="غير متوفر")
        tk.Label(left_metrics, textvariable=self.avg_source_length_var).grid(row=0, column=1, sticky='w', pady=2)
        tk.Label(left_metrics, text="متوسط طول النص المترجم:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky='w', pady=2)
        self.avg_target_length_var = tk.StringVar(value="غير متوفر")
        tk.Label(left_metrics, textvariable=self.avg_target_length_var).grid(row=1, column=1, sticky='w', pady=2)
        tk.Label(left_metrics, text="متوسط الخطأ التربيعي:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=2)
        self.mse_var = tk.StringVar(value="غير متوفر")
        tk.Label(left_metrics, textvariable=self.mse_var).grid(row=2, column=1, sticky='w', pady=2)
        tk.Label(right_metrics, text="نسبة الأحرف:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=2)
        self.char_ratio_var = tk.StringVar(value="غير متوفر")
        tk.Label(right_metrics, textvariable=self.char_ratio_var).grid(row=0, column=1, sticky='w', pady=2)
        tk.Label(right_metrics, text="عدد الترجمات:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky='w', pady=2)
        self.trans_count_var = tk.StringVar(value="0")
        tk.Label(right_metrics, textvariable=self.trans_count_var).grid(row=1, column=1, sticky='w', pady=2)
        tk.Label(right_metrics, text="متوسط الخطأ المطلق:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=2)
        self.mae_var = tk.StringVar(value="غير متوفر")
        tk.Label(right_metrics, textvariable=self.mae_var).grid(row=2, column=1, sticky='w', pady=2)
        examples_frame = tk.LabelFrame(results_frame, text="أمثلة الترجمة", padx=10, pady=10)
        examples_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.examples_tree = ttk.Treeview(examples_frame, columns=('الإنجليزية', 'العربية'), show='headings')
        self.examples_tree.heading('الإنجليزية', text='النص الإنجليزي')
        self.examples_tree.heading('العربية', text='الترجمة العربية')
        self.examples_tree.column('الإنجليزية', width=200)
        self.examples_tree.column('العربية', width=200)
        self.examples_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        x_scrollbar = ttk.Scrollbar(examples_frame, orient=tk.HORIZONTAL, command=self.examples_tree.xview)
        self.examples_tree.configure(xscrollcommand=x_scrollbar.set)
        x_scrollbar.pack(fill=tk.X)

    def setup_metrics_tab(self):
        main_frame = tk.Frame(self.metrics_tab, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            main_frame,
            text="مقاييس الترجمة",
            font=('Arial', 16, 'bold')
        ).pack(pady=10)
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        self.metrics_button = tk.Button(
            control_frame,
            text="إنشاء المقاييس",
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
            text="عارض مجموعة البيانات",
            font=('Arial', 16, 'bold')
        ).pack(pady=10)
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        self.load_dataset_btn = tk.Button(
            control_frame,
            text="تحميل مجموعة البيانات",
            command=self.browse_dataset,
            bg="#4CAF50",
            fg="white",
            padx=10,
            pady=5
        )
        self.load_dataset_btn.pack(side=tk.LEFT, padx=5)
        self.dataset_info = ttk.Label(main_frame, font=("Arial", 12))
        if self.dataset:
            self.dataset_info.config(text=f"تم تحميل {len(self.dataset)} زوج ترجمة من:\n{self.dataset_path}")
        else:
            self.dataset_info.config(text="لا توجد مجموعة بيانات محملة.")
        self.dataset_info.pack(pady=10)
        viewer_frame = tk.LabelFrame(main_frame, text="إدخالات مجموعة البيانات", padx=10, pady=10)
        viewer_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.dataset_tree = ttk.Treeview(viewer_frame, columns=('المعرف', 'الإنجليزية', 'العربية'), show='headings')
        self.dataset_tree.heading('المعرف', text='المعرف')
        self.dataset_tree.heading('الإنجليزية', text='النص الإنجليزي')
        self.dataset_tree.heading('العربية', text='الترجمة العربية')
        self.dataset_tree.column('المعرف', width=50)
        self.dataset_tree.column('الإنجليزية', width=200)
        self.dataset_tree.column('العربية', width=200)
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
            messagebox.showwarning("تحذير", "الرجاء إدخال نص للترجمة")
            return

        self.status_bar.config(text="جاري الترجمة...")
        self.translation_output.delete(1.0, tk.END)
        self.translation_output.insert(tk.END, "جاري الترجمة...")
        self.root.update()
        try:
            translated_text = translate_text_en_to_ar(text)
            self.translation_output.delete(1.0, tk.END)
            self.translation_output.insert(tk.END, translated_text)
            self.status_bar.config(text="تمت الترجمة")
            self.translations.append((text, translated_text))
        except Exception as e:
            self.translation_output.delete(1.0, tk.END)
            self.translation_output.insert(tk.END, f"خطأ في الترجمة: {str(e)}")
            self.status_bar.config(text="فشلت الترجمة")

    def save_translation(self):
        source_text = self.text_input.get("1.0", tk.END).strip()
        translated_text = self.translation_output.get("1.0", tk.END).strip()
        if not source_text or not translated_text:
            messagebox.showwarning("تحذير", "لا توجد ترجمة للحفظ")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("ملفات نصية", "*.txt"), ("جميع الملفات", "*.*")],
            title="حفظ الترجمة"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("النص الأصلي (الإنجليزية):\n")
                f.write(f"{source_text}\n\n")
                f.write("الترجمة (العربية):\n")
                f.write(f"{translated_text}\n")
            messagebox.showinfo("نجاح", f"تم حفظ الترجمة في:\n{file_path}")
        except Exception as e:
            messagebox.showerror("خطأ", f"فشل حفظ الترجمة: {str(e)}")

    def clear_fields(self):
        self.text_input.delete("1.0", tk.END)
        self.translation_output.delete("1.0", tk.END)

    def show_install_instructions(self):
        messagebox.showinfo(
            "تعليمات التثبيت",
            "لتمكين ميزات الترجمة، يرجى تثبيت الحزم التالية:\n\n"
            "pip install transformers sentencepiece\n\n"
            "بعد التثبيت، أعد تشغيل التطبيق."
        )

    def browse_dataset(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("ملفات نصية", "*.txt"), ("جميع الملفات", "*.*")],
            title="اختر ملف مجموعة البيانات"
        )
        if file_path:
            self.dataset_path = file_path
            self.load_dataset()
            if self.dataset:
                self.dataset_info.config(text=f"تم تحميل {len(self.dataset)} زوج ترجمة من:\n{self.dataset_path}")
                messagebox.showinfo("نجاح", f"تم تحميل {len(self.dataset)} زوج ترجمة من مجموعة البيانات")
                self.populate_dataset_tree()
                self.status_bar.config(text=f"تم تحميل {len(self.dataset)} زوج ترجمة من مجموعة البيانات")
                found_sample_button = False
                for widget in self.translation_tab.winfo_descendants():
                    if isinstance(widget, tk.Button) and widget.cget('text') == "استخدم عينة":
                        found_sample_button = True
                        break

                if not found_sample_button:
                    for widget in self.translation_tab.winfo_descendants():
                        if isinstance(widget, tk.Frame):
                            for child in widget.winfo_children():
                                if isinstance(child, tk.Button) and child.cget('text') == "تحميل مجموعة البيانات":
                                    self.sample_button = tk.Button(
                                        widget,
                                        text="استخدم عينة",
                                        command=self.use_dataset_sample,
                                        bg="#9C27B0",
                                        fg="white",
                                        padx=10,
                                        pady=5
                                    )
                                    self.sample_button.pack(side=tk.LEFT, padx=5)
                                    break

    def use_dataset_sample(self):
        if not self.dataset:
            messagebox.showwarning("تحذير", "لا توجد مجموعة بيانات محملة")
            return

        sample = random.choice(self.dataset)
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", sample[0])
        self.translation_output.delete("1.0", tk.END)

    def run_evaluation(self):
        if not self.translations:
            messagebox.showwarning("تحذير", "لا توجد ترجمات لتقييمها. يرجى ترجمة بعض النصوص أولاً.")
            return

        self.show_loading("جاري تقييم الترجمات...")
        try:
            self.evaluation_results = evaluate_translation(self.translations)
            self.avg_source_length_var.set(f"{self.evaluation_results['avg_source_length']:.2f}")
            self.avg_target_length_var.set(f"{self.evaluation_results['avg_target_length']:.2f}")
            self.char_ratio_var.set(f"{self.evaluation_results['char_ratio']:.2f}")
            self.trans_count_var.set(f"{len(self.translations)}")
            self.mse_var.set(f"{self.evaluation_results['mse']:.4f}")
            self.mae_var.set(f"{self.evaluation_results['mae']:.4f}")
            self.examples_tree.delete(*self.examples_tree.get_children())
            for source, target in self.evaluation_results['examples']:
                self.examples_tree.insert('', 'end', values=(source, target))
            self.evaluation_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.hide_loading()
            messagebox.showinfo("نجاح", "تم تقييم الترجمة بنجاح!")
            self.notebook.select(self.evaluation_tab)
        except Exception as e:
            self.hide_loading()
            error_message = f"حدث خطأ أثناء التقييم: {str(e)}"
            messagebox.showerror("خطأ التقييم", error_message)
            print(f"خطأ التقييم: {traceback.format_exc()}")
    def calculate_and_display_metrics(self):
        if not self.translations:
            messagebox.showwarning("تحذير", "لا توجد ترجمات لتحليلها. يرجى ترجمة بعض النصوص أولاً.")
            return

        self.show_loading("جاري حساب المقاييس...")
        try:
            total_translations = len(self.translations)
            source_lengths = [len(pair[0]) for pair in self.translations]
            target_lengths = [len(pair[1]) for pair in self.translations]
            mse_values = []
            mae_values = []
            for source, target in self.translations:
                mse = calculate_mse(source, target)
                mae = calculate_mae(source, target)
                mse_values.append(mse)
                mae_values.append(mae)
            self.metrics_info.config(
                text=f"إجمالي الترجمات: {total_translations}\n"
                     f"متوسط طول النص الأصلي: {np.mean(source_lengths):.2f} حرفًا\n"
                     f"متوسط طول النص المترجم: {np.mean(target_lengths):.2f} حرفًا\n"
                     f"متوسط نسبة الأحرف: {np.mean(target_lengths) / np.mean(source_lengths):.2f}\n"
                     f"متوسط الخطأ التربيعي (MSE): {np.mean(mse_values):.4f}\n"
                     f"متوسط الخطأ المطلق (MAE): {np.mean(mae_values):.4f}"
            )
            for widget in self.metrics_frame.winfo_children():
                widget.destroy()
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(221)
            ax1.bar(['الإنجليزية', 'العربية'], [np.mean(source_lengths), np.mean(target_lengths)])
            ax1.set_title('متوسط طول النص')
            ax1.set_ylabel('الأحرف')
            ax2 = fig.add_subplot(222)
            ax2.hist(source_lengths, alpha=0.5, label='الإنجليزية', bins=10)
            ax2.hist(target_lengths, alpha=0.5, label='العربية', bins=10)
            ax2.set_title('توزيع الطول')
            ax2.set_xlabel('الأحرف')
            ax2.set_ylabel('التكرار')
            ax2.legend()
            ax3 = fig.add_subplot(223)
            ax3.plot(range(len(mse_values)), mse_values, 'ro-')
            ax3.set_title('متوسط الخطأ التربيعي (MSE)')
            ax3.set_xlabel('فهرس الترجمة')
            ax3.set_ylabel('قيمة MSE')
            ax4 = fig.add_subplot(224)
            ax4.plot(range(len(mae_values)), mae_values, 'bo-')
            ax4.set_title('متوسط الخطأ المطلق (MAE)')
            ax4.set_xlabel('فهرس الترجمة')
            ax4.set_ylabel('قيمة MAE')
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self.metrics_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.hide_loading()
            self.notebook.select(self.metrics_tab)

        except Exception as e:
            self.hide_loading()
            error_message = f"حدث خطأ أثناء حساب المقاييس: {str(e)}"
            messagebox.showerror("خطأ المقاييس", error_message)
            print(f"خطأ المقاييس: {traceback.format_exc()}")

    def save_evaluation_results(self):
        if self.evaluation_results is None:
            messagebox.showwarning("تحذير", "لا توجد نتائج تقييم للحفظ. يرجى تشغيل التقييم أولاً.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("ملفات نصية", "*.txt"), ("جميع الملفات", "*.*")],
            title="حفظ نتائج التقييم"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"نتائج تقييم الترجمة - {self.evaluation_timestamp}\n\n")
                f.write(f"عدد الترجمات: {len(self.translations)}\n")
                f.write(f"متوسط طول النص الأصلي: {self.evaluation_results['avg_source_length']:.2f} حرفًا\n")
                f.write(f"متوسط طول النص المترجم: {self.evaluation_results['avg_target_length']:.2f} حرفًا\n")
                f.write(f"نسبة الأحرف (العربية/الإنجليزية): {self.evaluation_results['char_ratio']:.2f}\n")
                f.write(f"متوسط الخطأ التربيعي (MSE): {self.evaluation_results['mse']:.4f}\n")
                f.write(f"متوسط الخطأ المطلق (MAE): {self.evaluation_results['mae']:.4f}\n\n")
                f.write("أمثلة الترجمة:\n")
                for i, (source, target) in enumerate(self.evaluation_results['examples']):
                    f.write(f"المثال {i + 1}:\n")
                    f.write(f"  الإنجليزية: {source}\n")
                    f.write(f"  العربية: {target}\n\n")
            messagebox.showinfo("نجاح", f"تم حفظ نتائج التقييم في:\n{file_path}")
        except Exception as e:
            messagebox.showerror("خطأ", f"فشل حفظ النتائج: {str(e)}")

    def show_loading(self, message="جاري التحميل..."):
        if hasattr(self, 'loading_window') and self.loading_window is not None:
            self.loading_window.destroy()

        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.title("جاري المعالجة")
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
    app = TranslationApp(root)
    root.mainloop()