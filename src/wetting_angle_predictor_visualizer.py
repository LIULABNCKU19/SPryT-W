#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wetting Angle Prediction and Visualization System
Combines Gaussian Kernel Ridge Regression model with wetting angle visualization
Allows users to visualize wetting angle predictions for different material compositions,
substrate materials, and temperatures

Author: LIULAB@NCKU
Version: 2025.09.15Ver
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import math
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
import joblib

# Set matplotlib backend
import matplotlib
matplotlib.use('TkAgg')

# Suppress Tk deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'

class WettingAnglePredictorVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Wetting Angle Prediction Visualizer")
        self.root.geometry("1400x800")
        
        # Set font for Chinese characters
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Initialize model and substrate data
        self.model = None
        self.substrate_data = None
        self.feature_columns = None
        self.is_model_loaded = False
        
        # Create GUI interface
        self.create_widgets()
        
        # Load model and substrate data
        self.load_model_and_data()
        
        # Initialize plots
        self.setup_plot()
        
        # Initial plot
        self.update_display()
    
    def create_widgets(self):
        """
        Create and configure GUI widgets
        """
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left control panel
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Material composition control panel
        composition_frame = ttk.LabelFrame(left_frame, text="Material Composition (Weight %) ", padding=10)
        composition_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create material composition sliders
        self.composition_vars = {}
        composition_materials = [
            ('Bi_wt', 'Bismuth (Bi)', 0, 20),
            ('Zn_wt', 'Zinc (Zn)', 0, 9),
            ('Ag_wt', 'Silver (Ag)', 0, 4),
            ('Cu_wt', 'Copper (Cu)', 0, 0.8),
            ('Al_wt', 'Aluminum (Al)', 0, 0.45),
            ('Sb_wt', 'Antimony (Sb)', 0, 0.5),
            ('In_wt', 'Indium (In)', 0, 25),
            ('Ni_wt', 'Nickel (Ni)', 0, 0.3)
        ]
        
        for material, label, min_val, max_val in composition_materials:
            ttk.Label(composition_frame, text=f"{label}:").pack(anchor=tk.W)
            
            # Create horizontal frame containing slider and input box
            input_frame = ttk.Frame(composition_frame)
            input_frame.pack(fill=tk.X, pady=(0, 2))
            
            var = tk.DoubleVar(value=0.0)
            scale = ttk.Scale(input_frame, from_=min_val, to=max_val, 
                            variable=var, orient=tk.HORIZONTAL,
                            command=lambda v, m=material: self.on_composition_change(m))
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
            
            # Add input box
            entry = ttk.Entry(input_frame, textvariable=var, width=8)
            entry.pack(side=tk.RIGHT)
            entry.bind('<Return>', lambda e, m=material: self.on_composition_entry_change(m))
            entry.bind('<FocusOut>', lambda e, m=material: self.on_composition_entry_change(m))
            
            label_widget = ttk.Label(composition_frame, text=f"{label} = 0.0%")
            label_widget.pack(anchor=tk.W, pady=(0, 5))
            
            self.composition_vars[material] = {
                'var': var,
                'label': label_widget,
                'scale': scale,
                'entry': entry,
                'display_name': label,
                'min_val': min_val,
                'max_val': max_val
            }
        
        # Substrate material and temperature control panel
        condition_frame = ttk.LabelFrame(left_frame, text="Substrate Material and Reflow Temperature", padding=10)
        condition_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Substrate material selection
        ttk.Label(condition_frame, text="Substrate Material:").pack(anchor=tk.W)
        self.substrate_var = tk.StringVar(value="Cu")
        self.substrate_combo = ttk.Combobox(condition_frame, textvariable=self.substrate_var, 
                                          state="readonly", width=20)
        self.substrate_combo.pack(fill=tk.X, pady=(0, 10))
        self.substrate_combo.bind('<<ComboboxSelected>>', self.on_substrate_change)
        
        # Temperature control
        ttk.Label(condition_frame, text="Reflow Temperature (°C):").pack(anchor=tk.W)
        
        # Create horizontal frame for temperature control
        temp_frame = ttk.Frame(condition_frame)
        temp_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.temperature_var = tk.DoubleVar(value=250.0)
        self.temperature_scale = ttk.Scale(temp_frame, from_=220, to=300, 
                                         variable=self.temperature_var, orient=tk.HORIZONTAL,
                                         command=self.on_condition_change)
        self.temperature_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Add temperature input box
        self.temperature_entry = ttk.Entry(temp_frame, textvariable=self.temperature_var, width=8)
        self.temperature_entry.pack(side=tk.RIGHT)
        self.temperature_entry.bind('<Return>', self.on_temperature_entry_change)
        self.temperature_entry.bind('<FocusOut>', self.on_temperature_entry_change)
        
        self.temperature_label = ttk.Label(condition_frame, text="Reflow Temperature = 250.0 °C")
        self.temperature_label.pack(anchor=tk.W)
        
        # Prediction results display panel
        prediction_frame = ttk.LabelFrame(left_frame, text="Prediction Results", padding=10)
        prediction_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.prediction_label = ttk.Label(prediction_frame, text="Wetting Angle: --°", 
                                        font=("Arial", 14, "bold"))
        self.prediction_label.pack(anchor=tk.W, pady=5)
        
        self.interpretation_label = ttk.Label(prediction_frame, text="", 
                                            wraplength=200)
        self.interpretation_label.pack(anchor=tk.W, pady=5)
        
        
        # Preset composition buttons
        preset_frame = ttk.LabelFrame(left_frame, text="Preset Compositions", padding=10)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        
        presets = [
            ("Pure Sn", {'Sn_wt': 100, 'Bi_wt': 0, 'Zn_wt': 0, 'Ag_wt': 0, 'Cu_wt': 0, 
                        'Al_wt': 0, 'Sb_wt': 0, 'In_wt': 0, 'Ni_wt': 0}),
            ("Sn-Ag Alloy", {'Sn_wt': 96.5, 'Bi_wt': 0, 'Zn_wt': 0, 'Ag_wt': 3.5, 'Cu_wt': 0,
                           'Al_wt': 0, 'Sb_wt': 0, 'In_wt': 0, 'Ni_wt': 0}),
            ("Bi Alloy", {'Sn_wt': 80, 'Bi_wt': 20, 'Zn_wt': 0, 'Ag_wt': 0, 'Cu_wt': 0,
                         'Al_wt': 0, 'Sb_wt': 0, 'In_wt': 0, 'Ni_wt': 0})
        ]
        
        for name, composition in presets:
            ttk.Button(preset_frame, text=name, 
                      command=lambda c=composition: self.set_preset_composition(c)).pack(side=tk.LEFT, padx=2)
        
        # Right plot display area
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def load_model_and_data(self):
        """
        Load the trained ML model and substrate data
        """
        try:
            # Define file paths (relative to current script location)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # script_dir is /path/to/SPrT-W/src, so project_root should be /path/to/SPrT-W
            project_root = script_dir  # The src directory is at the same level as data
            substrate_path = os.path.join(os.path.dirname(project_root), 'data', 'substrate.xlsx')
            model_path = os.path.join(os.path.dirname(project_root), 'data', 'gaussian_kernel_ridge_model.pkl')
            
            # Load substrate data
            if os.path.exists(substrate_path):
                self.substrate_data = pd.read_excel(substrate_path)
                substrate_names = self.substrate_data.iloc[:, 0].tolist()
                self.substrate_combo['values'] = substrate_names
                print(f"Loaded {len(substrate_names)} substrate materials: {substrate_names}")
            else:
                messagebox.showerror("Error", f"substrate.xlsx file not found at {substrate_path}")
                return
            
            # Load trained model
            if os.path.exists(model_path):
                # Create model instance and load
                self.model = GaussianKernelRidgeRegression()
                self.model.load_model(model_path)
                self.feature_columns = self.model.feature_columns
                self.is_model_loaded = True
                print(f"Model loaded, feature columns: {self.feature_columns}")
            else:
                messagebox.showerror("Error", f"Trained model file not found at {model_path}")
                return
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {str(e)}")
    
    def setup_plot(self):
        """Set plot properties"""
        # Left plot: Wetting angle diagram
        self.ax1.set_xlim(-3, 3)
        self.ax1.set_ylim(-1, 3)
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title('Wetting Angle Diagram', fontsize=12, fontweight='bold')
        
        # Right plot: Material composition and prediction results
        self.ax2.set_title('Material Composition and Prediction Results', fontsize=12, fontweight='bold')
        self.ax2.axis('off')
    
    def get_current_input_data(self):
        """Get current input data"""
        # Get material composition
        input_data = {}
        for material in self.composition_vars:
            value = self.composition_vars[material]['var'].get()
            input_data[material] = value
        
        # Get temperature
        input_data['Temperature'] = self.temperature_var.get()
        
        # Get substrate material properties
        substrate_name = self.substrate_var.get()
        if self.substrate_data is not None:
            substrate_row = self.substrate_data[self.substrate_data.iloc[:, 0] == substrate_name]
            if not substrate_row.empty:
                input_data['FirstIonizationEnergy'] = substrate_row.iloc[0]['FirstIonizationEnergy']
                input_data['BCCmagmom'] = substrate_row.iloc[0]['BCCmagmom']
                input_data['MeltingT'] = substrate_row.iloc[0]['MeltingT']
        
        return input_data
    
    def predict_wetting_angle(self):
        """Predict wetting angle - using pre-trained model"""
        if not self.is_model_loaded:
            return None
        
        try:
            # Get input data
            input_data = self.get_current_input_data()
            
            # Create test data DataFrame
            test_data = {
                'Bi_wt': [input_data['Bi_wt']],
                'Zn_wt': [input_data['Zn_wt']],
                'Ag_wt': [input_data['Ag_wt']],
                'Cu_wt': [input_data['Cu_wt']],
                'Al_wt': [input_data['Al_wt']],
                'Sb_wt': [input_data['Sb_wt']],
                'In_wt': [input_data['In_wt']],
                'Ni_wt': [input_data['Ni_wt']],
                'Temperature': [input_data['Temperature']],
                'FirstIonizationEnergy': [input_data['FirstIonizationEnergy']],
                'BCCmagmom': [input_data['BCCmagmom']],
                'MeltingT': [input_data['MeltingT']]
            }
            test_df = pd.DataFrame(test_data)
            
            # Get test features
            X_test = test_df[self.feature_columns].values
            
            # Use pre-trained model for prediction (no need for training.xlsx)
            prediction = self.model.predict(X_test)[0]
            
            return prediction
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None
    
    def calculate_geometry(self, theta_deg, size_factor=1.0):
        """Calculate geometric parameters based on wetting angle and size factor"""
        # Convert predicted wetting angle to display angle
        actual_theta_deg = theta_deg
        actual_theta_rad = math.radians(actual_theta_deg)
        
        # Set droplet radius (adjusted by size factor)
        radius = 1.5 * size_factor
        
        # Calculate contact line radius
        contact_radius = radius * math.sin(actual_theta_rad)
        
        # Calculate droplet center position
        center_x = 0
        center_y = radius * math.cos(actual_theta_rad)
        
        return center_x, center_y, radius, contact_radius, actual_theta_rad, actual_theta_deg
    
    def draw_wetting_angle_diagram(self, theta_deg):
        """Draw wetting angle diagram"""
        self.ax1.clear()
        self.ax1.set_xlim(-3, 3)
        self.ax1.set_ylim(-1, 3)
        self.ax1.set_aspect('equal')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title('Wetting Angle Diagram', fontsize=12, fontweight='bold')
        
        # Draw solid surface
        solid_surface = patches.Rectangle((-3, -0.5), 6, 0.5, 
                                        facecolor='lightgray', edgecolor='black', 
                                        linewidth=2)
        self.ax1.add_patch(solid_surface)
        
        # Add substrate material label on solid surface
        substrate_name = self.substrate_var.get()
        self.ax1.text(0, -0.25, f'Substrate: {substrate_name}', 
                     fontsize=12, fontweight='bold', ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Calculate geometric parameters
        center_x, center_y, radius, contact_radius, theta_rad, _ = self.calculate_geometry(theta_deg)
        
        # Draw droplet
        droplet_circle = patches.Circle((center_x, center_y), radius, 
                                      facecolor='lightblue', edgecolor='darkblue', 
                                      linewidth=2, alpha=0.8)
        self.ax1.add_patch(droplet_circle)
        
        # Cover y<0 part with solid surface
        solid_overlay = patches.Rectangle((-3, -1), 6, 1, 
                                        facecolor='lightgray', edgecolor='black', 
                                        linewidth=0, alpha=1.0)
        self.ax1.add_patch(solid_overlay)
        
        # Remove contact line display
        
        # Remove wetting angle annotation
        
        # Remove legend as there are no elements to display
    
    def draw_composition_and_results(self, input_data, prediction, visualization_angle):
        """Draw material composition and prediction results"""
        self.ax2.clear()
        self.ax2.set_title('Material Composition and Prediction Results', fontsize=12, fontweight='bold')
        self.ax2.axis('off')
        
        # Display material composition (remove Test Data text)
        y_pos = 0.95
        
        composition_text = ""
        total_composition = 0
        
        for material in self.composition_vars:
            value = input_data[material]
            if value > 0:
                composition_text += f"{self.composition_vars[material]['display_name']}: {value:.1f}%\n"
                total_composition += value
        
        # Calculate remaining Sn
        remaining_sn = 100 - total_composition
        if remaining_sn > 0:
            composition_text += f"Remaining Tin (Sn): {remaining_sn:.1f}%\n"
        elif remaining_sn < 0:
            composition_text += f"Remaining Tin (Sn): 0.0% (Total > 100%)\n"
        else:
            composition_text += f"Remaining Tin (Sn): 0.0%\n"
        
        if not composition_text.strip():
            composition_text = "Pure Tin (Sn)"
        
        self.ax2.text(0.1, y_pos, composition_text, fontsize=11, ha='left', va='top')
        y_pos -= len(composition_text.split('\n')) * 0.06 + 0.08
        
        # Display substrate material and temperature
        substrate_info = f"Substrate: {self.substrate_var.get()}\nReflow Temperature: {input_data['Temperature']:.1f} °C"
        self.ax2.text(0.1, y_pos, substrate_info, fontsize=11, ha='left', va='top')
        y_pos -= 0.12
        
        # Display prediction results (moved down to leave space for elements)
        self.ax2.text(0.1, y_pos, "Prediction Results:", fontsize=14, fontweight='bold', ha='left')
        y_pos -= 0.08
        
        # Numerical display: original predicted value
        result_text = f"Wetting Angle: {prediction:.1f}°"
        self.ax2.text(0.1, y_pos, result_text, fontsize=16, fontweight='bold', 
                     ha='left', color='red')
        
        # Add powered by LIULAB@NCKU text in bottom right corner
        self.ax2.text(0.98, 0.02, 'powered by LIULAB@NCKU', 
                     fontsize=10, ha='right', va='bottom', 
                     style='italic', color='gray', alpha=0.7)
        
        # Remove hydrophilicity text display
        
        # Remove substrate property information display
    
    def update_display(self):
        """
        Update all display elements with current parameters
        """
        # Predict wetting angle
        prediction = self.predict_wetting_angle()
        
        if prediction is not None:
            # Numerical display: directly show model predicted wetting angle value
            self.prediction_label.config(text=f"Wetting Angle: {prediction:.1f}°")
            
            # Remove wetting angle explanation display
            self.interpretation_label.config(text="")
            
            # Get current input data
            input_data = self.get_current_input_data()
            
            # Visualization display: use (180 - wetting angle) for drawing
            visualization_angle = 180 - prediction
            self.draw_wetting_angle_diagram(visualization_angle)
            
            # Draw material composition and prediction results (numerical display uses original value, graphical display uses visualization angle)
            self.draw_composition_and_results(input_data, prediction, visualization_angle)
        
        # Refresh canvas
        self.canvas.draw()
    
    def on_composition_change(self, material):
        """Callback when material composition changes"""
        value = self.composition_vars[material]['var'].get()
        self.composition_vars[material]['label'].config(text=f"{self.composition_vars[material]['display_name']} = {value:.1f}%")
        self.update_display()
    
    def on_composition_entry_change(self, material):
        """Callback when material composition input box changes"""
        try:
            value = float(self.composition_vars[material]['var'].get())
            min_val = self.composition_vars[material]['min_val']
            max_val = self.composition_vars[material]['max_val']
            
            # Limit to valid range
            value = max(min_val, min(max_val, value))
            self.composition_vars[material]['var'].set(value)
            
            # Update label
            self.composition_vars[material]['label'].config(text=f"{self.composition_vars[material]['display_name']} = {value:.1f}%")
            self.update_display()
        except ValueError:
            # If input is invalid, restore original value
            pass
    
    def on_substrate_change(self, event):
        """Substrate material change callback"""
        print(f"Substrate material changed to: {self.substrate_var.get()}")
        self.update_display()
    
    def on_condition_change(self, value):
        """Temperature change callback"""
        temp = float(value)
        self.temperature_label.config(text=f"Reflow Temperature = {temp:.1f} °C")
        self.update_display()
    
    def on_temperature_entry_change(self, event):
        """Temperature input box change callback"""
        try:
            temp = float(self.temperature_var.get())
            # Limit to valid range (220-300°C)
            temp = max(220.0, min(300.0, temp))
            self.temperature_var.set(temp)
            
            # Update label
            self.temperature_label.config(text=f"Reflow Temperature = {temp:.1f} °C")
            self.update_display()
        except ValueError:
            # If input is invalid, restore original value
            pass
    
    def set_preset_composition(self, composition):
        """Set preset composition"""
        for material, value in composition.items():
            if material in self.composition_vars:
                self.composition_vars[material]['var'].set(value)
                self.composition_vars[material]['label'].config(
                    text=f"{self.composition_vars[material]['display_name']} = {value:.1f}%")
        
        self.update_display()

# Simplified model class (for loading saved model)
class GaussianKernelRidgeRegression:
    def __init__(self, alpha=0.00212, gamma=0.008685):
        self.alpha = alpha
        self.gamma = gamma
        self.model = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def load_model(self, filepath):
        """Load saved model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_fitted = True
            print(f"Model loaded from {filepath}")
            print(f"Model parameters: alpha={self.alpha}, gamma={self.gamma}")
            print(f"Feature columns: {self.feature_columns}")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def predict(self, X):
        """Predict"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        # Standardize input data before prediction
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions

def main():
    """Main function"""
    root = tk.Tk()
    app = WettingAnglePredictorVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
