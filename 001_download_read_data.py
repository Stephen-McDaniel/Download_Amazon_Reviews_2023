###########################################################################################################
# Download a category of Amazon Reviews from the McAuley-Lab, 
#     at https://cseweb.ucsd.edu/~jmcauley/
#
# This program downloads user reviews (VERY LARGE)
#     for the course at 
#     https://www.peakpython.com/rethinking-python-pandas-10x-speed-flexibility-with-python-ibis
# Create download directories and change the core_path variable
# 
# Stephen McDaniel at https://PeakPython.com
# 2024-NOV-19
#
# License: MIT License
###########################################################################################################
 
import requests
import pandas as pd
import gzip
import json
from pathlib import Path

# find the categories at https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
category='Health_and_Household'
category='Magazine_Subscriptions'

core_path = '/root/pc_01_pandasibis/110_McAuley_Amazon_Data/amazon_reviews/'
output_csv = core_path + '/processed/' + category + '.csv'
input_file = core_path + '/raw/' + category + '.jsonl.gz'

# Replace with the actual URL of the dataset
output_file = category + '.jsonl.gz'
local_path = core_path + '/raw/'
url = 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/' + output_file

response = requests.get(url, stream=True)
response.raise_for_status()  # Check for request errors

with open(local_path + '/' + output_file, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:  # Filter out keep-alive chunks
            f.write(chunk)

print("Download completed.")



def parse_json_chunks(input_file, chunksize, start_row=0, max_rows=10000000):
    """
    Parse JSON file in chunks and add row numbers.
    
    Args:
        input_file (str): Path to input gzipped JSON file
        chunksize (int): Number of rows to process at once
        start_row (int): Starting row number for the chunk
    
    Yields:
        pd.DataFrame: DataFrame with row numbers and parsed JSON data
    """
    if start_row < max_rows:
        with gzip.open(input_file, 'rt', encoding='utf-8') as f:
            chunk = []
            current_row = start_row
            
            for line in f:
                try:
                    record = json.loads(line)
                    record['row_number'] = current_row + 1  # Add row number to each record
                    chunk.append(record)
                    current_row += 1
                    
                    if len(chunk) >= chunksize:
                        yield pd.DataFrame(chunk)
                        chunk = []
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {current_row + 1}: {e}")
                    continue
            
            # Yield the last chunk if any records remain
            if chunk:
                yield pd.DataFrame(chunk)



def convert_json_to_csv(input_file, output_csv, chunksize=10**6, max_rows = 10000000):
    """
    Convert JSON file to CSV with row numbers.
    
    Args:
        input_file (str): Path to input gzipped JSON file
        output_csv (str): Path to output CSV file
        chunksize (int): Number of rows to process at once
    """
    # Convert paths to Path objects
    input_path = Path(input_file)
    output_path = Path(output_csv)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Track total rows processed
    total_rows = 0
    first_chunk = True
    
    print(f"Converting {input_path} to {output_path}")
    print(f"Processing in chunks of {chunksize:,} rows")
    
    try:
        for df_chunk in parse_json_chunks(input_path, chunksize, total_rows, max_rows):
            if total_rows < max_rows:
                # Ensure row_number is the first column
                columns = ['row_number'] + [col for col in df_chunk.columns if col != 'row_number']
                df_chunk = df_chunk[columns]
                
                # Write the chunk to CSV
                df_chunk.to_csv(
                    output_path,
                    mode='w' if first_chunk else 'a',
                    index=False,
                    header=first_chunk,
                    encoding='utf-8'
                )
                
                # Update progress
                chunk_size = len(df_chunk)
                total_rows += chunk_size
                print(f"Processed {total_rows:,} rows (chunk size: {chunk_size:,})")
                
                first_chunk = False
            
        print(f"\nConversion completed successfully!")
        print(f"Total rows processed: {total_rows:,}")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError during conversion: {str(e)}")
        raise



if __name__ == "__main__":
    # Example usage
    
    convert_json_to_csv(
        input_file=input_file,
        output_csv=output_csv,
        chunksize=1_000_000  # Process 1 million rows at a time
    )
