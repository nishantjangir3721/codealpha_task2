import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------------------
# Load dataset
# -------------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, names=columns, na_values='?')

# Preprocessing
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)
data = data.dropna()

# Features & Target
X = data.drop('target', axis=1)
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

# Create heart.csv if it doesn't exist
if not os.path.exists('heart.csv'):
    empty_df = pd.DataFrame(columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
        'Diagnosis', 'Confidence',
        'Logistic_Regression_Prediction', 'Logistic_Regression_Confidence',
        'SVM_Prediction', 'SVM_Confidence',
        'Random_Forest_Prediction', 'Random_Forest_Confidence',
        'XGBoost_Prediction', 'XGBoost_Confidence'
    ])
    empty_df.to_csv('heart.csv', index=False)

# -------------------------------
# GUI Application
# -------------------------------
class MedicalPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CardioCare Predictor")
        self.root.geometry("1200x750")
        self.root.configure(bg='#f0f8ff')

        # Custom style
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f8ff')
        self.style.configure('TLabel', background='#f0f8ff', font=('Helvetica', 10))
        self.style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'), foreground='#2e5984')
        self.style.configure('TButton', font=('Helvetica', 10), padding=5)
        self.style.configure('TNotebook', background='#f0f8ff')
        self.style.configure('TNotebook.Tab', font=('Helvetica', 10, 'bold'), padding=[10, 5])
        self.style.configure('Treeview', font=('Helvetica', 10))
        self.style.configure('Treeview.Heading', font=('Helvetica', 10, 'bold'), anchor='center')

        # Header
        header_frame = ttk.Frame(root)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(header_frame, text="CardioCare Disease Predictor", style='Header.TLabel').pack(side=tk.LEFT, padx=10)

        # Notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Tabs
        self.input_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.records_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.input_tab, text='Patient Information')
        self.notebook.add(self.results_tab, text='Prediction Results')
        self.notebook.add(self.records_tab, text='Saved Records')

        # Build input form
        self.create_input_form()
        self.create_results_tab()
        self.create_records_tab()

        # Store the default CSV file for saving
        self.save_file_path = 'heart.csv'

    # -------------------------------
    # Input Form
    # -------------------------------
    def create_input_form(self):
        form_frame = ttk.Frame(self.input_tab)
        form_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        sections = [
            ("Personal Information", ['age', 'sex']),
            ("Symptoms", ['cp', 'exang']),
            ("Vital Signs", ['trestbps', 'thalach', 'oldpeak']),
            ("Test Results", ['chol', 'fbs', 'restecg', 'slope', 'ca', 'thal'])
        ]

        self.entries = {}
        current_row = 0

        field_details = {
            'age': {'label': 'Age (years)', 'type': 'int'},
            'sex': {'label': 'Gender', 'type': 'option', 'options': ['Female', 'Male']},
            'cp': {'label': 'Chest Pain Type', 'type': 'option',
                   'options': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']},
            'exang': {'label': 'Exercise Induced Angina', 'type': 'option', 'options': ['No', 'Yes']},
            'trestbps': {'label': 'Resting Blood Pressure (mm Hg)', 'type': 'int'},
            'thalach': {'label': 'Maximum Heart Rate Achieved', 'type': 'int'},
            'oldpeak': {'label': 'ST Depression (Exercise)', 'type': 'float'},
            'chol': {'label': 'Serum Cholesterol (mg/dl)', 'type': 'int'},
            'fbs': {'label': 'Fasting Blood Sugar > 120 mg/dl', 'type': 'option', 'options': ['No', 'Yes']},
            'restecg': {'label': 'Resting ECG', 'type': 'option',
                        'options': ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy']},
            'slope': {'label': 'Slope of Peak Exercise ST Segment', 'type': 'option',
                      'options': ['Upsloping', 'Flat', 'Downsloping']},
            'ca': {'label': 'Major Vessels Colored (0-3)', 'type': 'option',
                   'options': ['0', '1', '2', '3']},
            'thal': {'label': 'Thalassemia', 'type': 'option',
                     'options': ['Normal', 'Fixed Defect', 'Reversible Defect']}
        }

        for section, fields in sections:
            ttk.Label(form_frame, text=section, style='Header.TLabel').grid(
                row=current_row, column=0, columnspan=2, pady=(10, 5), sticky='w')
            current_row += 1

            for field in fields:
                details = field_details[field]
                ttk.Label(form_frame, text=details['label']).grid(
                    row=current_row, column=0, padx=5, pady=2, sticky='e')
                if details['type'] == 'option':
                    var = tk.StringVar()
                    cb = ttk.Combobox(form_frame, textvariable=var,
                                      values=details['options'], state='readonly')
                    cb.grid(row=current_row, column=1, padx=5, pady=2, sticky='ew')
                    cb.current(0)
                    self.entries[field] = var
                else:
                    entry = ttk.Entry(form_frame)
                    entry.grid(row=current_row, column=1, padx=5, pady=2, sticky='ew')
                    entry.insert(0, "0")  # default value
                    self.entries[field] = entry
                current_row += 1

        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=current_row, column=0, columnspan=2, pady=20)
        ttk.Button(button_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Clear", command=self.clear_form).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Select Save File", command=self.select_save_file).pack(side=tk.LEFT, padx=10)

    # -------------------------------
    # Results Tab
    # -------------------------------
    def create_results_tab(self):
        results_frame = ttk.Frame(self.results_tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        ttk.Label(results_frame, text="Prediction Results", style='Header.TLabel').pack()
        self.results_display = ttk.Frame(results_frame)
        self.results_display.pack(fill=tk.BOTH, expand=True, pady=10)
        self.update_results_display([])

    def update_results_display(self, results):
        for widget in self.results_display.winfo_children():
            widget.destroy()

        if not results:
            ttk.Label(self.results_display, text="No prediction results available. Please submit patient data.",
                      style='Header.TLabel').pack(expand=True)
            return

        overall_frame = ttk.Frame(self.results_display)
        overall_frame.pack(fill=tk.X, pady=10)
        avg_confidence = np.mean([r['confidence'] for r in results])
        diagnosis = "POSITIVE for Heart Disease" if avg_confidence >= 0.5 else "NEGATIVE for Heart Disease"
        color = "#d9534f" if avg_confidence >= 0.5 else "#5cb85c"

        ttk.Label(overall_frame, text="Overall Diagnosis:", style='Header.TLabel').pack(side=tk.LEFT, padx=5)
        ttk.Label(overall_frame, text=diagnosis, font=('Helvetica', 12, 'bold'),
                  foreground=color).pack(side=tk.LEFT, padx=5)

        # Confidence meter
        meter_frame = ttk.Frame(self.results_display)
        meter_frame.pack(fill=tk.X, pady=10)
        fig, ax = plt.subplots(figsize=(8, 0.8))
        ax.set_facecolor('#f0f8ff')
        fig.patch.set_facecolor('#f0f8ff')
        ax.barh(0, avg_confidence * 100, color=color, height=0.5)
        ax.set_xlim(0, 100)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(avg_confidence * 100 + 2, 0, f"{avg_confidence * 100:.1f}%", va='center', color=color, fontsize=10)
        ax.set_title('Confidence Level', fontsize=10)
        canvas = FigureCanvasTkAgg(fig, master=meter_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.X)
        plt.close(fig)

        # Per-model results
        models_frame = ttk.Frame(self.results_display)
        models_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        for i, result in enumerate(results):
            model_frame = ttk.Frame(models_frame, relief='groove', borderwidth=1)
            model_frame.grid(row=i // 2, column=i % 2, padx=5, pady=5, sticky='nsew')
            ttk.Label(model_frame, text=result['model'], font=('Helvetica', 10, 'bold')).pack(anchor='w', padx=5,
                                                                                              pady=2)
            pred_color = "#d9534f" if result['prediction'] == 1 else "#5cb85c"
            pred_text = "Positive" if result['prediction'] == 1 else "Negative"
            ttk.Label(model_frame, text=f"Prediction: {pred_text}", foreground=pred_color).pack(anchor='w', padx=5,
                                                                                                pady=2)
            conf_frame = ttk.Frame(model_frame)
            conf_frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT)
            ttk.Progressbar(conf_frame, orient='horizontal', length=100,
                            mode='determinate', value=result['confidence'] * 100).pack(side=tk.LEFT, padx=5)
            ttk.Label(conf_frame, text=f"{result['confidence'] * 100:.1f}%").pack(side=tk.LEFT)
        models_frame.columnconfigure(0, weight=1)
        models_frame.columnconfigure(1, weight=1)

    # -------------------------------
    # Records Tab
    # -------------------------------
    def create_records_tab(self):
        frame = ttk.Frame(self.records_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Saved Patient Records", style="Header.TLabel").pack(pady=5)

        # Create a frame for the Treeview and scrollbars
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        # Vertical scrollbar
        y_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Horizontal scrollbar
        x_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Create the Treeview
        self.tree = ttk.Treeview(tree_frame, yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Configure the scrollbars
        y_scroll.config(command=self.tree.yview)
        x_scroll.config(command=self.tree.xview)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Load Records", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear Records", command=self.clear_records).pack(side=tk.LEFT, padx=5)

    def select_save_file(self):
        """Allow user to select a CSV file for saving predictions."""
        file_path = filedialog.asksaveasfilename(
            title="Select CSV File to Save Predictions",
            filetypes=[("CSV files", "*.csv")],
            defaultextension=".csv",
            initialfile="heart.csv"
        )
        if file_path:
            self.save_file_path = file_path
            messagebox.showinfo("Success", f"Predictions will be saved to {os.path.basename(file_path)}.")

    def load_data(self):
        """Handle loading data from a CSV file via file dialog."""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return  # User canceled the dialog

        try:
            # Validate the CSV structure
            expected_columns = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
                'Diagnosis', 'Confidence',
                'Logistic_Regression_Prediction', 'Logistic_Regression_Confidence',
                'SVM_Prediction', 'SVM_Confidence',
                'Random_Forest_Prediction', 'Random_Forest_Confidence',
                'XGBoost_Prediction', 'XGBoost_Confidence'
            ]
            df = pd.read_csv(file_path)

            # Check if all expected columns are present
            if not all(col in df.columns for col in expected_columns):
                missing_cols = [col for col in expected_columns if col not in df.columns]
                messagebox.showerror("Error", f"Selected CSV is missing columns: {', '.join(missing_cols)}")
                return

            # If loading in records tab, update Treeview
            if self.notebook.index(self.notebook.select()) == 2:  # Records tab
                self.load_records(file_path)
            else:  # Input tab
                # Load last record into input form (if available)
                if not df.empty:
                    last_record = df.iloc[-1]
                    for field in self.entries:
                        if field in last_record:
                            value = str(last_record[field])
                            if isinstance(self.entries[field], tk.StringVar):
                                self.entries[field].set(value)
                            else:
                                self.entries[field].delete(0, tk.END)
                                self.entries[field].insert(0, value)
                    messagebox.showinfo("Success", f"Loaded data from {os.path.basename(file_path)} into input form.")
                else:
                    messagebox.showinfo("Info", "Selected CSV is empty.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")

    def load_records(self, file_path=None):
        """Load records into Treeview from a CSV file."""
        # Clear existing treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Clear existing columns
        for col in self.tree["columns"]:
            self.tree.column(col, width=0, minwidth=0)
            self.tree.heading(col, text="")

        # Use default heart.csv if no file_path provided
        if not file_path:
            file_path = self.save_file_path
            if not os.path.exists(file_path):
                messagebox.showinfo("Info", f"No records found in {os.path.basename(file_path)}.")
                return

        try:
            df = pd.read_csv(file_path)

            # Setup columns
            self.tree["columns"] = list(df.columns)
            for col in df.columns:
                max_width = max(
                    [len(str(col))] +
                    [len(str(val)) for val in df[col].astype(str)]
                )
                width = min(max_width * 8, 150)
                self.tree.heading(col, text=col, anchor='center')
                self.tree.column(col, width=width, minwidth=50, anchor='center', stretch=tk.YES)

            # Insert rows
            for _, row in df.iterrows():
                self.tree.insert("", tk.END, values=list(row))

            messagebox.showinfo("Success", f"Records loaded from {os.path.basename(file_path)}.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load records: {str(e)}")

    def clear_records(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        messagebox.showinfo("Info", "Records display cleared.")

    # -------------------------------
    # Utility
    # -------------------------------
    def clear_form(self):
        for field, widget in self.entries.items():
            if isinstance(widget, tk.StringVar):
                defaults = {
                    'sex': 'Female', 'cp': 'Typical Angina', 'exang': 'No', 'fbs': 'No',
                    'restecg': 'Normal', 'slope': 'Upsloping', 'ca': '0', 'thal': 'Normal'
                }
                widget.set(defaults.get(field, ''))
            elif isinstance(widget, tk.Entry):
                widget.delete(0, tk.END)
                widget.insert(0, "0")
        self.notebook.select(self.input_tab)

    def predict(self):
        try:
            input_mapping = {
                'sex': {'Female': 0, 'Male': 1},
                'cp': {'Typical Angina': 0, 'Atypical Angina': 1,
                       'Non-anginal Pain': 2, 'Asymptomatic': 3},
                'exang': {'No': 0, 'Yes': 1},
                'fbs': {'No': 0, 'Yes': 1},
                'restecg': {'Normal': 0, 'ST-T Wave Abnormality': 1,
                            'Left Ventricular Hypertrophy': 2},
                'slope': {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2},
                'ca': {'0': 0, '1': 1, '2': 2, '3': 3},
                'thal': {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
            }

            input_data = []
            original_inputs = {}

            for field in columns[:-1]:
                if field in self.entries:
                    value = self.entries[field].get()
                    if not value:
                        raise ValueError(f"Field {field} cannot be empty")
                    original_inputs[field] = value
                    if field in input_mapping:
                        mapped_value = input_mapping[field].get(value, -1)
                        if mapped_value == -1:
                            raise ValueError(f"Invalid selection for {field}")
                        input_data.append(mapped_value)
                    else:
                        try:
                            input_data.append(float(value) if field == 'oldpeak' else int(value))
                        except ValueError:
                            raise ValueError(f"Invalid value for {field}: {value}")

            # Scale input
            input_df = pd.DataFrame([input_data], columns=X.columns)
            input_scaled = scaler.transform(input_df)

            results = []
            for name, model in models.items():
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0][1]
                results.append({'model': name, 'prediction': prediction, 'confidence': proba})

            # Overall
            avg_confidence = np.mean([r['confidence'] for r in results])
            diagnosis = "Positive" if avg_confidence >= 0.5 else "Negative"

            # Save to CSV
            save_dict = original_inputs.copy()
            save_dict["Diagnosis"] = diagnosis
            save_dict["Confidence"] = round(avg_confidence, 3)

            # Add model-specific results
            model_name_mapping = {
                "Logistic Regression": "Logistic_Regression",
                "SVM": "SVM",
                "Random Forest": "Random_Forest",
                "XGBoost": "XGBoost"
            }

            for r in results:
                model_key = model_name_mapping[r['model']]
                save_dict[f"{model_key}_Prediction"] = "Positive" if r['prediction'] == 1 else "Negative"
                save_dict[f"{model_key}_Confidence"] = round(r['confidence'], 3)

            # Convert to DataFrame
            save_df = pd.DataFrame([save_dict])

            # Ensure the save file has the correct structure
            expected_columns = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
                'Diagnosis', 'Confidence',
                'Logistic_Regression_Prediction', 'Logistic_Regression_Confidence',
                'SVM_Prediction', 'SVM_Confidence',
                'Random_Forest_Prediction', 'Random_Forest_Confidence',
                'XGBoost_Prediction', 'XGBoost_Confidence'
            ]
            save_df = save_df.reindex(columns=expected_columns)

            # Save to CSV
            try:
                file_exists = os.path.isfile(self.save_file_path)
                save_df.to_csv(self.save_file_path, mode='a', index=False, header=not file_exists)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save to {os.path.basename(self.save_file_path)}: {str(e)}")
                return

            # Update UI
            self.update_results_display(results)
            self.notebook.select(self.results_tab)
            messagebox.showinfo("Success", f"Patient data saved to {os.path.basename(self.save_file_path)} successfully!")

        except Exception as e:
            messagebox.showerror("Input Error", f"Please check your inputs:\n{str(e)}")

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalPredictorApp(root)
    root.mainloop()