import pandas as pd
import dataframe_image as dfi

def save_top_five_as_image(csv_file_path, output_image_path):
    # 1. Load the CSV file
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print("File not found.")
        return

    # 2. Get top 5 records
    top_five = df.head(5)

    # 3. Save as image
    # Using dataframe_image to export the table directly
    dfi.export(top_five, output_image_path, max_cols=-1)
    print(f"Top 5 records saved to {output_image_path}")



# Run the function
save_top_five_as_image('./Week3/full_preprocessed.csv', 'full_preprocessed.png')
