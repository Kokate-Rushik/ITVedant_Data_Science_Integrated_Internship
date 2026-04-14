import nbformat as nbf

def py_to_ipynb(input_py_path, output_ipynb_path):
    # 1. Read the .py file content
    with open(input_py_path, 'r', encoding='utf-8') as f:
        code_content = f.read()

    # 2. Split the code into sections based on your delimiter
    # This creates a list of code blocks
    code_segments = code_content.split('%$%')

    # 3. Initialize a new notebook object (v4)
    nb = nbf.v4.new_notebook()

    # 4. Create a new code cell for each segment
    # .strip() removes leading/trailing empty lines for a cleaner notebook
    nb['cells'] = [nbf.v4.new_code_cell(segment.strip()) 
                   for segment in code_segments if segment.strip()]

    # 5. Write the final notebook file
    with open(output_ipynb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"Successfully converted {input_py_path} to {output_ipynb_path}")

# Run the function
py_to_ipynb('./test/my_script.py', 'output_notebook.ipynb')
