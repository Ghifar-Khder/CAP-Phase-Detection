import os
import tensorflow as tf
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
class ModelPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Predictor")
        self.root.geometry("650x750")  # Bigger interface
        self.root.configure(bg="#1E3A5F")  # Set blue background color

        # Default Paths
        self.model_path = r"models\narco_model.keras"
        self.dataset_path = r"data\splitted-data\narco\narco_test_set.csv"
        
        # Load Default Model
        self.model = tf.keras.models.load_model(self.model_path)

        # Frames for Paths (Model & Dataset)
        self.model_path_frame = tk.Frame(root, bg="#1E3A5F")
        self.dataset_path_frame = tk.Frame(root, bg="#1E3A5F")
        self.model_path_frame.pack_forget()
        self.dataset_path_frame.pack_forget()

        # Model Path Toggle Button (Bigger)
        self.btn_toggle_model = tk.Button(
            root, text="⏷ Model Path", command=self.toggle_model_path, width=20, 
            bg="#3B82F6", fg="white", font=("Arial", 14, "bold"), height=2
        )
        self.btn_toggle_model.pack(padx=10, pady=5)

        # Dataset Path Toggle Button (Bigger)
        self.btn_toggle_dataset = tk.Button(
            root, text="⏷ Dataset Path", command=self.toggle_dataset_path, width=20, 
            bg="#3B82F6", fg="white", font=("Arial", 14, "bold"), height=2
        )
        self.btn_toggle_dataset.pack(padx=10, pady=5)

        # Create Model Path Fields
        self.entry_model_path = tk.Entry(self.model_path_frame, width=80, font=("Arial", 12))
        self.entry_model_path.insert(0, self.model_path)
        self.entry_model_path.pack(pady=5, padx=10)

        self.btn_browse_model = tk.Button(
            self.model_path_frame, text="📂 Browse Model", command=self.select_model, 
            bg="#60A5FA", fg="white", font=("Arial", 12, "bold"), width=20
        )
        self.btn_browse_model.pack(pady=5)

        # Create Dataset Path Fields
        self.entry_manual_path = tk.Entry(self.dataset_path_frame, width=80, font=("Arial", 12))
        self.entry_manual_path.insert(0, self.dataset_path)
        self.entry_manual_path.pack(pady=5, padx=10)

        self.btn_manual_select = tk.Button(
            self.dataset_path_frame, text="📂 Load Dataset", command=self.load_manual_dataset, 
            bg="#60A5FA", fg="white", font=("Arial", 12, "bold"), width=20
        )
        self.btn_manual_select.pack(pady=5)

        self.btn_select = tk.Button(
            self.dataset_path_frame, text="📂 Browse Dataset", command=self.select_dataset, 
            bg="#60A5FA", fg="white", font=("Arial", 12, "bold"), width=20
        )
        self.btn_select.pack(pady=5)

        # Label for Dataset
        self.label_dataset = tk.Label(root, text=f"Dataset: {os.path.basename(self.dataset_path)}", 
                                      font=("Arial", 14, "bold"), bg="#1E3A5F", fg="white")
        self.label_dataset.pack(pady=5)

        self.label_samples = tk.Label(root, text="Total Samples: Not Loaded", 
                                      font=("Arial", 14, "bold"), bg="#1E3A5F", fg="white")
        self.label_samples.pack(pady=5)

        self.label_input = tk.Label(root, text="Enter Sample Row Number:", 
                                    font=("Arial", 12), bg="#1E3A5F", fg="white")
        self.label_input.pack(pady=5)

        self.entry_row = tk.Entry(root, font=("Arial", 12))
        self.entry_row.pack(pady=5)

        self.btn_predict = tk.Button(
            root, text="🔍 Predict", command=self.make_prediction, bg="#2563EB", fg="white", 
            font=("Arial", 14, "bold"), width=20, height=2
        )
        self.btn_predict.pack(pady=5)

        self.label_result = tk.Label(root, text="Prediction Result:", font=("Arial", 14, "bold"), 
                                     bg="#1E3A5F", fg="white")
        self.label_result.pack(pady=10)

        # Load the default dataset
        self.load_dataset(self.dataset_path)

    def toggle_model_path(self):
        if self.model_path_frame.winfo_ismapped():
            self.model_path_frame.pack_forget()
            self.btn_toggle_model.config(text="⏷ Model Path")
        else:
            self.model_path_frame.pack(before=self.btn_toggle_dataset, padx=10, pady=5)
            self.btn_toggle_model.config(text="⏶ Model Path")

    def toggle_dataset_path(self):
        if self.dataset_path_frame.winfo_ismapped():
            self.dataset_path_frame.pack_forget()
            self.btn_toggle_dataset.config(text="⏷ Dataset Path")
        else:
            self.dataset_path_frame.pack(before=self.label_dataset, padx=10, pady=5)
            self.btn_toggle_dataset.config(text="⏶ Dataset Path")

    def select_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Keras Model", "*.keras")])
        if file_path:
            self.model_path = file_path
            self.entry_model_path.delete(0, tk.END)
            self.entry_model_path.insert(0, file_path)
            self.model = tf.keras.models.load_model(self.model_path)
            messagebox.showinfo("Success", "Model loaded successfully!")

    def select_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.dataset_path = file_path
            self.entry_manual_path.delete(0, tk.END)
            self.entry_manual_path.insert(0, file_path)
            self.load_dataset(file_path)

    def load_manual_dataset(self):
        file_path = self.entry_manual_path.get().strip()
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "File does not exist! Check the path and try again.")
            return
        self.load_dataset(file_path)

    def load_dataset(self, filepath):
        try:
            data = pd.read_csv(filepath).values
            self.x_data = data[:, :-1]
            self.y_data = data[:, -1]
            self.x_data = self.x_data.reshape(self.x_data.shape[0], self.x_data.shape[1], 1)
            self.dataset_name = os.path.basename(filepath)
            self.label_dataset.config(text=f"Dataset: {self.dataset_name}")
            self.label_samples.config(text=f"Total Samples: {len(self.x_data)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")

    def make_prediction(self):
        try:
            row_number = int(self.entry_row.get())
            x_row = self.x_data[row_number]
            y_actual = int(self.y_data[row_number])
            y_pred_prob = self.model.predict(x_row.reshape(1, -1, 1))[0][0] * 100
            y_pred = 1 if y_pred_prob > 50 else 0
            correctness = "✅ Correct!!" if y_actual == y_pred else "❌ Wrong!!"
            result_text = f"📄 Dataset: {self.dataset_name}\n Row number: {row_number}\n🔹 Actual: {y_actual}\n Predicted: {y_pred}\n📊 Prediction Probability: {abs(100-(y_pred*100)-y_pred_prob):.1f}%\n{correctness}"
            self.label_result.config(text=result_text)
            
            # Plot the EEG signal
            self.plot_signal(x_row, y_actual,row_number)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid row number!\n{str(e)}")

    def plot_signal(self, signal, y_actual,row_number):
        plt.figure(figsize=(8, 3))
        plt.plot(signal, linestyle='-', marker='', color='b')
        plt.xlabel("Sample Number")
        plt.ylabel("EEG Signal")
        plt.title(f"{'A signal' if y_actual == 1 else 'B signal'} - Row: {row_number}")
        plt.grid(True)
        plt.show()

root = tk.Tk()
app = ModelPredictorApp(root)
root.mainloop()
