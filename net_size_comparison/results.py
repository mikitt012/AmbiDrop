import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

directory="/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/net_size_comparison"

csv_files = glob.glob(os.path.join(directory, "metrics_params_*.csv"))

if not csv_files:
    print("No CSV files found matching 'metrics_params_*.csv'")

for file in csv_files:

    def generate_final_report(csv_path):
        # 1. Load data and round to 2 decimals
        df = pd.read_csv(csv_path)
        df = df.round(2)
        params = df['Model_Params'].iloc[0]
        
        # 2. Clean Array Names: remove "_preprocessed"
        df['Array_Type'] = df['Array_Type'].str.replace('_preprocessed', '', case=False)
        
        # 3. Split and Calculate Averages
        def get_section(dataset_name):
            subset = df[df['Dataset'] == dataset_name].drop(columns=['Model_Params', 'Dataset'])
            numeric_cols = subset.select_dtypes(include=[np.number]).columns
            avg_vals = subset[numeric_cols].mean().round(2)
            
            avg_row = {col: (avg_vals[col] if col in numeric_cols else 'AVERAGE') for col in subset.columns}
            return pd.concat([subset, pd.DataFrame([avg_row])], ignore_index=True)

        test_table = get_section('Test_Arrays')
        train_table = get_section('Train_Arrays')

        # 4. Create the Figure
        fig, ax = plt.subplots(figsize=(16, 20)) # Increased width for array names
        ax.axis('off')
        
        # Header: Model Parameters
        plt.text(0.5, 0.98, f"Model Architecture: {params:,} Parameters", 
                fontsize=22, weight='bold', ha='center', transform=ax.transAxes)

        def draw_styled_table(df_plot, start_y, title, color):
            # Section Title
            ax.text(0.5, start_y, title, fontsize=16, weight='bold', ha='center', 
                    color='white', bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.5'),
                    transform=ax.transAxes)
            
            # Define multi-level headers manually for the table
            headers = ['Array Name', 'Noisy', 'Enhanced', 'Noisy', 'Enhanced', 'Noisy', 'Enhanced']
            data = [headers] + df_plot.values.tolist()
            
            # Table position with wider first column (0.4 for Array Name)
            # colWidths=[Array, SI-SDR(N), SI-SDR(E), PESQ(N), PESQ(E), STOI(N), STOI(E)]
            col_widths = [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            
            table_height = len(data) * 0.025
            table = ax.table(cellText=data, loc='center', cellLoc='center', colWidths=col_widths,
                            bbox=[0.0, start_y - table_height - 0.04, 1.0, table_height])
            
            table.auto_set_font_size(False)
            table.set_fontsize(12)

            # Apply Styling
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor('#ABB2B9')
                if row == 0: # Headers
                    cell.set_facecolor('#EBEDEF')
                    cell.set_text_props(weight='bold')
                elif row == len(data) - 1: # Average
                    cell.set_facecolor('#FCF3CF')
                    cell.set_text_props(weight='bold')
                if col == 0: # Left-align Array Names for better room
                    cell.set_text_props(ha='left')
            
            # Add Metric Group Labels
            y_label = start_y - 0.035
            ax.text(0.5, y_label, "SI-SDR", weight='bold', ha='center', transform=ax.transAxes, fontsize=12)
            ax.text(0.7, y_label, "PESQ", weight='bold', ha='center', transform=ax.transAxes, fontsize=12)
            ax.text(0.9, y_label, "STOI", weight='bold', ha='center', transform=ax.transAxes, fontsize=12)

            return start_y - table_height - 0.12

        # Draw Sections
        next_y = draw_styled_table(test_table, 0.92, "TEST ARRAYS PERFORMANCE", "#E67E22")
        draw_styled_table(train_table, next_y, "TRAIN ARRAYS PERFORMANCE", "#27AE60")

        plt.savefig(f"net_size_comparison/Final_Report_{params}.png", dpi=300, bbox_inches='tight')
        plt.show()

    generate_final_report(file)

def plot_model_comparison(directory="."):
    all_data = []

    # 1. Iterate through all metrics CSV files
    csv_files = glob.glob(os.path.join(directory, "metrics_params_*.csv"))
    
    for file in csv_files:
        df = pd.read_csv(file)
        params = df['Model_Params'].iloc[0]
        
        # Calculate means for each dataset type
        for dataset in ['Train_Arrays', 'Test_Arrays']:
            subset = df[df['Dataset'] == dataset]
            all_data.append({
                'Params': params,
                'Dataset': dataset,
                'SI-SDR': subset['SI_SDR_Enhanced'].mean(),
                'PESQ': subset['PESQ_Enhanced'].mean(),
                'STOI': subset['STOI_Enhanced'].mean()
            })

    # 2. Create summary DataFrame
    summary_df = pd.DataFrame(all_data)
    
    # Sort by Params: Largest to Smallest
    summary_df = summary_df.sort_values(by='Params', ascending=False)
    
    # Get unique sorted params for X-axis labels (as strings to prevent linear scaling)
    sorted_params = summary_df['Params'].unique()
    x_labels = [f"{p:,}" for p in sorted_params]
    x_indices = range(len(sorted_params))

    metrics = ['SI-SDR', 'PESQ', 'STOI']
    colors = {'Train_Arrays': '#27AE60', 'Test_Arrays': '#E67E22'} # Green and Orange

    # 3. Generate 3 Plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    plt.subplots_adjust(hspace=0.4)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for dataset in ['Train_Arrays', 'Test_Arrays']:
            # Filter data for the specific line
            plot_data = summary_df[summary_df['Dataset'] == dataset]
            
            # Use marker to show exact model points
            ax.plot(x_indices, plot_data[metric], label=dataset.replace('_', ' '), 
                    color=colors[dataset], marker='o', linewidth=2, markersize=8)

        # Formatting
        ax.set_title(f"Model Comparison: {metric} (Higher is Better)", fontsize=16, weight='bold')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel("Model Parameters (Largest → Smallest)", fontsize=12)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    plt.tight_layout()
    plt.savefig("net_size_comparison/model_performance_comparison.png", dpi=300)
    plt.show()

# plot_model_comparison(directory="/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/net_size_comparison")

def plot_combined_trends(directory="."):
    results = []
    
    # 1. Gather data from all metrics CSVs
    csv_files = glob.glob(os.path.join(directory, "metrics_params_*.csv"))
    
    if not csv_files:
        print("No CSV files found matching 'metrics_params_*.csv'")
        return

    for file in csv_files:
        df = pd.read_csv(file)
        params = df['Model_Params'].iloc[0]
        
        # Calculate the mean across ALL rows (merging Train and Test performance)
        results.append({
            'Params': params,
            'SI-SDR': df['SI_SDR_Enhanced'].mean(),
            'PESQ': df['PESQ_Enhanced'].mean(),
            'STOI': df['STOI_Enhanced'].mean()
        })

    # 2. Sort from Largest to Smallest Parameters
    summary_df = pd.DataFrame(results).sort_values(by='Params', ascending=False)
    
    # 3. Plotting setup
    metrics = ['SI-SDR', 'PESQ', 'STOI']
    colors = ['#2980B9', '#C0392B', '#8E44AD'] # Distinct colors for each plot
    x_labels = [f"{int(p):,}" for p in summary_df['Params']]
    x_indices = range(len(summary_df))

    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    plt.subplots_adjust(hspace=0.4)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        y_vals = summary_df[metric]
        
        # Plot single mean line
        ax.plot(x_indices, y_vals, color=colors[i], marker='o', 
                linewidth=3, markersize=10, markerfacecolor='white', markeredgewidth=2)

        # Annotate each point with its value for clarity
        for x, y in zip(x_indices, y_vals):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', weight='bold', fontsize=10)

        # Title and Labels
        ax.set_title(f"Mean Performance: {metric} (Combined Train/Test)", fontsize=16, weight='bold', pad=15)
        ax.set_ylabel(metric, fontsize=12, weight='bold')
        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.xlabel("Model Parameters (Largest → Smallest)", fontsize=14, weight='bold', labelpad=15)
    plt.tight_layout()
    
    output_path = "net_size_comparison/Model_Performance_Trends_Combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success: Plot saved as {output_path}")

# plot_combined_trends(directory="/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/net_size_comparison")

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

def plot_combined_si_sdri(directory="."):
    results = []
    
    # 1. Gather data from all metrics CSVs
    csv_files = glob.glob(os.path.join(directory, "metrics_params_*.csv"))
    
    if not csv_files:
        print("No CSV files found matching 'metrics_params_*.csv'")
        return

    for file in csv_files:
        df = pd.read_csv(file)
        params = df['Model_Params'].iloc[0]
        
        # Calculate SI-SDR improvement (SI-SDRi = Enhanced - Noisy)
        # Taking the mean across all rows (Train and Test)
        if 'SI_SDR_Enhanced' in df.columns and 'SI_SDR_Noisy' in df.columns:
            si_sdri = (df['SI_SDR_Enhanced'] - df['SI_SDR_Noisy']).mean()
        else:
            # Fallback to absolute enhanced value if Noisy column is missing
            si_sdri = df['SI_SDR_Enhanced'].mean()
        
        results.append({
            'Params': params,
            'SI-SDRi': si_sdri
        })

    # 2. Sort from Largest to Smallest Parameters
    summary_df = pd.DataFrame(results).sort_values(by='Params', ascending=False)
    
    # 3. Plotting setup
    x_labels = [f"{int(p):,}" for p in summary_df['Params']]
    x_indices = range(len(summary_df))

    plt.figure(figsize=(10, 6))
    y_vals = summary_df['SI-SDRi']
    
    # Plot mean trend line
    plt.plot(x_indices, y_vals, color='#2980B9', marker='o', 
             linewidth=3, markersize=10, markerfacecolor='white', markeredgewidth=2)

    # # Annotate points with values
    # for x, y in zip(x_indices, y_vals):
    #     plt.annotate(f'{y:.2f} dB', (x, y), textcoords="offset points", 
    #                 xytext=(0,18), ha='center', weight='bold', fontsize=10)

    # Title and Labels
    # plt.title("Model Robustness: SI-SDR Improvement vs. Network Size", fontsize=14, weight='bold', pad=20)
    plt.ylabel("SI-SDRi [dB]", fontsize=20)
    plt.xlabel("Number of Parameters", fontsize=20, labelpad=10)

    # ax.set_xlabel("Sample Count (Log Scale)", fontsize=23, labelpad=15)
    # ax.set_ylabel("Input SI-SDR Bin [dB]", fontsize=23, labelpad=15)
    
    # Make tick numbers bigger
    # ax.tick_params(axis='both', which='major', labelsize=20)
    
    plt.xticks(x_indices, x_labels, rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.ylim([10,13.5]) # Add padding for annotations
    
    plt.tight_layout()
    
    output_path = "net_size_comparison/SI_SDRi_Trend_no_values2.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success: Plot saved as {output_path}")

plot_combined_si_sdri(directory="/Users/mikitatarjitzky/Documents/AmbiDrop Code/AmbiDrop/net_size_comparison")