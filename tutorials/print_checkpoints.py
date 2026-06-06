import pandas as pd
import os

def main():
    file_path = "tutorials/wandb_runs/hasse_metrics_checkpoints.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: Checkpoint file not found at {file_path}")
        return

    # Load the checkpoint CSV
    df = pd.read_csv(file_path)
    
    if df.empty:
        print("The checkpoint file is empty.")
        return

    # Iterate through each dataset entry
    for _, row in df.iterrows():
        dataset_name = row['dataset_name']
        print(f"\n{'='*80}")
        print(f" DATASET: {dataset_name}")
        print(f"{'='*80}")
        
        # Group and print metrics
        # The columns are formatted as nbhd_metric_stat
        current_nbhd = None
        
        # Sort columns to keep neighborhoods together
        metric_cols = sorted([col for col in df.columns if col != 'dataset_name'])
        
        for col in metric_cols:
            val = row[col]
            if pd.isna(val):
                continue
                
            # Detect neighborhood change for better formatting
            nbhd = col.split('_')[0] if 'adjacency' in col or 'incidence' in col else "General"
            if nbhd != current_nbhd:
                current_nbhd = nbhd
                print(f"\n  [Neighborhood: {current_nbhd}]")
            
            # Print metric name (truncated nbhd) and value
            metric_display = col.replace(f"{current_nbhd}_", "")
            print(f"    {metric_display:40} : {val:10.4f}")

    print(f"\nTotal datasets processed: {len(df)}")

if __name__ == "__main__":
    main()
