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
    dfi.export(top_five, output_image_path)
    print(f"Top 5 records saved to {output_image_path}")

# Example Usage:
# Create a dummy CSV for demonstration
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
        'Score': [85, 90, 78, 92, 88, 76]}
pd.DataFrame(data).to_csv('data.csv', index=False)

# Run the function
save_top_five_as_image('data.csv', 'top_five.png')
