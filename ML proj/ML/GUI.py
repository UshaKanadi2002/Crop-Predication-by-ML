import tkinter as tk
from tkinter import ttk, messagebox, Canvas
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load dataset
df = pd.read_csv('reduced_dataset.csv')
df.fillna(0, inplace=True)

# Encoding season values
season_map = {'Kharif': 1, 'Rabi': 2, 'Summer': 3, 'Winter': 4, 'Whole Year': 5}
df['season_encoded'] = df['season'].map(season_map).fillna(0)

# Define features and target
features = ['area', 'production', 'season_encoded', 'state_name']
target = 'crop_type'

X = df[features]
y = df[target]

# One-hot encoding for categorical data (state_name)
ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
X_encoded = ohe.fit_transform(X[['state_name']])
encoded_df = pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out())

# Final feature set
X_final = pd.concat([
    X[['area', 'production', 'season_encoded']].reset_index(drop=True),
    encoded_df.reset_index(drop=True)
], axis=1)

# Scaling numerical values
scaler = StandardScaler()
X_final[['area', 'production', 'season_encoded']] = scaler.fit_transform(
    X_final[['area', 'production', 'season_encoded']]
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_final, y)

# Tkinter GUI setup
root = tk.Tk()
root.title("Crop Prediction System with Advanced Visualizations")
root.geometry("1100x850")
root.configure(bg='#e3f2fd')

# Scrollable frame
canvas = Canvas(root)
scroll_y = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
frame = ttk.Frame(canvas)

frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

canvas.create_window((0, 0), window=frame, anchor="nw")
canvas.configure(yscrollcommand=scroll_y.set)
canvas.pack(side="left", fill="both", expand=True)
scroll_y.pack(side="right", fill="y")

# Input Section
input_frame = ttk.LabelFrame(frame, text="🌱 Enter Crop Details", padding=25)
input_frame.pack(pady=20, padx=30, fill="x")

def create_input(label_text):
    label = ttk.Label(input_frame, text=label_text)
    label.pack(pady=8)
    entry = ttk.Entry(input_frame, width=35)
    entry.pack()
    return entry

area_entry = create_input("Area (in hectares):")
production_entry = create_input("Production (in tonnes):")

season_label = ttk.Label(input_frame, text="Season:")
season_label.pack(pady=8)
season_combo = ttk.Combobox(input_frame, values=list(season_map.keys()), width=33)
season_combo.pack()

state_label = ttk.Label(input_frame, text="State:")
state_label.pack(pady=8)
state_combo = ttk.Combobox(input_frame, values=df['state_name'].unique().tolist(), width=33)
state_combo.pack()

# Table for crop types
table_frame = ttk.Frame(frame)
table_frame.pack(pady=10)

tree = ttk.Treeview(table_frame, columns=("Crop Type", "Crops"), show='headings', height=7)
tree.heading("Crop Type", text="Crop Type")
tree.heading("Crops", text="Crop Name")
tree.column("Crop Type", width=200)
tree.column("Crops", width=600)
tree.pack(side="left", fill="y")

scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")

# Graph visualization frame
plot_frame = ttk.Frame(frame)
plot_frame.pack(pady=10)

result_label = ttk.Label(frame, text="", font=('Verdana', 16, 'bold'))
result_label.pack(pady=10)

# Function to visualize crop comparison
def visualize_comparison(prediction):
    try:
        related_crops = df[df['crop_type'] == prediction]['crop_name'].value_counts()
        
        if related_crops.empty:
            messagebox.showinfo("Info", "No related crops found for visualization.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))  # Increased figure size
        related_crops.plot(kind='bar', color='#4caf50', ax=ax)

        ax.set_ylabel('Occurrences', fontsize=12)
        ax.set_xlabel('Crop Names', fontsize=12)
        ax.set_title(f'Comparison of Crops in {prediction} Category', fontsize=14, fontweight='bold')

        plt.xticks(rotation=45, ha="right", fontsize=10)  # Rotate labels
        plt.subplots_adjust(bottom=0.35, left=0.1, right=0.95, top=0.9)  # Adjust layout
        
        for widget in plot_frame.winfo_children():
            widget.destroy()
        
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
    
    except Exception as e:
        messagebox.showerror("Error", f"Error in visualization: {str(e)}")

# Function to update table
def update_table(prediction):
    tree.delete(*tree.get_children())  # Clear old data

    related_crops = df[df['crop_type'] == prediction][['crop_type', 'crop_name']].drop_duplicates()
    
    if related_crops.empty:
        tree.insert("", "end", values=("No Data", "No related crops found"))
        return

    for _, row in related_crops.iterrows():
        tree.insert("", "end", values=(row['crop_type'], row['crop_name']))

# Crop prediction function
def predict_crop():
    try:
        area = float(area_entry.get())
        production = float(production_entry.get())
        season = season_map.get(season_combo.get(), 0)
        state = state_combo.get()

        if not state:
            messagebox.showwarning("Warning", "Please select a state.")
            return

        input_df = pd.DataFrame({'area': [area], 'production': [production], 'season_encoded': [season]})
        encoded_input = ohe.transform(np.array([[state]]))
        encoded_input_df = pd.DataFrame(encoded_input, columns=ohe.get_feature_names_out())

        final_input = pd.concat([input_df.reset_index(drop=True), encoded_input_df], axis=1)
        final_input[['area', 'production', 'season_encoded']] = scaler.transform(
            final_input[['area', 'production', 'season_encoded']]
        )

        prediction = model.predict(final_input)[0]
        result_label.config(text=f"🌾 Predicted Crop: {prediction}", foreground='#1b5e20')
        
        update_table(prediction)  # Update table with predicted crops
        visualize_comparison(prediction)  # Show graph
    
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values for area and production.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Clear input fields
def clear_inputs():
    area_entry.delete(0, tk.END)
    production_entry.delete(0, tk.END)
    season_combo.set('')
    state_combo.set('')
    result_label.config(text='')
    tree.delete(*tree.get_children())  # Clear table
    for widget in plot_frame.winfo_children():
        widget.destroy()

# Buttons
predict_btn = ttk.Button(frame, text="Predict Crop", command=predict_crop)
predict_btn.pack(pady=12)

clear_btn = ttk.Button(frame, text="Clear Inputs", command=clear_inputs)
clear_btn.pack(pady=5)

# Run Tkinter loop
root.mainloop()
