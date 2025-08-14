import json
import tkinter as tk
from tkinter import ttk, Message
from Types import Predictor, ROI

with open('./config.json', mode='r', encoding='utf-8') as f:
    config = json.loads(f.read())

def update_roi_description(_=None):
    """Update ROI description when combobox selection changes"""
    selected_roi = roi_combo_box.get()
    roi_desc.config(text=config_roi['descriptions'][selected_roi])

def update_model_description(_=None):
    """Update Model description when combobox selection changes"""
    selected_model = model_combo_box.get()
    model_desc.config(text=config_predictor['descriptions'][selected_model])

def run_prediction():
    """Function to execute when Run button is clicked"""
    result_label.config(text="Running prediction for " +
                        model_combo_box.get() + " on " +
                        roi_combo_box.get() + " region")

root = tk.Tk()
root.title("Demographic Predictive Models")

###################### Configs ######################
config_roi = config['roi']
config_predictor = config['predictor']

###################### Default Values ######################
default_roi = ROI.PFC.name
default_model = Predictor.BRAIN_AGE.name

###################### Region of Interest ######################
roi_label = tk.Label(root, text="Pick region of interest: ")
roi_label.pack(pady=10)

# Create a Combobox widget
roi_combo_box = ttk.Combobox(
    root,
    values=config_roi['types'],
    state='readonly'
)
roi_combo_box.pack(pady=5)
roi_combo_box.set(default_roi)

# ROI description
roi_desc = Message(
    root,
    text=config_roi['descriptions'][default_roi],
    width=400
)
roi_desc.pack()

###################### Model Options ######################
model_label = tk.Label(root, text="Pick Model: ")
model_label.pack(pady=10)

# Create a Combobox widget
model_combo_box = ttk.Combobox(
    root,
    values=config_predictor['types'],
    state='readonly'
)
model_combo_box.pack(pady=5)
model_combo_box.set(default_model)

# Model description
model_desc = Message(
    root,
    text=config_predictor['descriptions'][default_model],
    width=400
)
model_desc.pack()

###################### Reporting ######################
run_button = ttk.Button(
    root,
    text="Run Prediction",
    command=run_prediction,
    style='Accent.TButton'
)
run_button.pack(pady=10, ipadx=10, ipady=5)

result_label = tk.Label(root, text="", fg='green')
result_label.pack()

# Bind events to update descriptions
roi_combo_box.bind("<<ComboboxSelected>>", update_roi_description)
model_combo_box.bind("<<ComboboxSelected>>", update_model_description)

root.mainloop()