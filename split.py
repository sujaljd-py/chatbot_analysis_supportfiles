import pandas as pd
import os
from tqdm import tqdm

def split_csv(input_file, output_dir, num_splits):
    # Read the CSV file
    df = pd.read_csv(input_file,encoding="iso-8859-1")

    # Extract total rows and columns
    total_rows = len(df)
    header = df.columns
    rows_per_file = total_rows // num_splits  # Base row count per file
    remainder = total_rows % num_splits  # Handle leftover rows

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    start_idx = 0
    files_created = []
    total_written_rows = 0  # To track total rows written

    # Use tqdm for progress tracking
    for i in tqdm(range(num_splits), desc="Splitting CSV", unit="file"):
        # Calculate end index, distributing remainder rows first
        end_idx = start_idx + rows_per_file + (1 if i < remainder else 0)

        # Slice the dataframe
        split_df = df.iloc[start_idx:end_idx]

        # Define output file path
        output_file = os.path.join(output_dir, f"Thousand{i+1}.csv")
        split_df.to_csv(output_file, index=False, columns=header)

        # Track row count
        row_count = len(split_df)
        total_written_rows += row_count

        # Store created file path
        files_created.append((output_file, row_count))

        # Update start index for next batch
        start_idx = end_idx

    # Validation
    print(f"\n✅ CSV split into {num_splits} files successfully!\n")
    for file, count in files_created:
        print(f"{file}: {count} rows")

    print(f"\n🔍 Original Rows: {total_rows}, Written Rows: {total_written_rows}")

    # Final Check
    if total_rows == total_written_rows:
        print("✅ All rows have been correctly distributed.")
    else:
        print("❌ Warning: Row mismatch! Check for missing or extra rows.")

# Example usage
input_csv = "C:/Users/Sujal Jadhv/Downloads/CleanedSet1.csv"  # Change this to your input CSV path
output_folder = "C:/Chatbot/RawSet1"  # Change this to your desired output directory
num_files = 16  # Number of splits

split_csv(input_csv, output_folder, num_files)
