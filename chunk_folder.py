import os
import subprocess

def get_pdf_files(input_dir):
    return [f for f in os.listdir(input_dir) if f.endswith('.pdf')]

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    for line in process.stdout:
        print(line.decode('utf-8').strip())

    stderr_output = process.stderr.read().decode('utf-8')
    if stderr_output:
        print(f"Error running command: {command}\n{stderr_output}")

def main():
    # Step 1: Get list of input PDFs
    input_dir = 'pdf_extraction_cell/data/input/'
    pdf_files = get_pdf_files(input_dir)

    # Step 2: Change directory to `./pdf_extraction_cell`
    os.chdir('pdf_extraction_cell')

    # Step 3: Run the extraction command
    extraction_command = 'python main.py --input data/input/ --output data/output -p -tp -ip'
    run_command(extraction_command)

    # Step 4: Change back to the parent directory
    os.chdir('..')

    # Step 5: Process each input PDF file
    for pdf_file in pdf_files:
        prefix = os.path.splitext(pdf_file)[0]
        chunking_command = f'python -m chunking_cell.main --prefix "{prefix}" --document-id "{prefix}"'
        run_command(chunking_command)

    # Step 6: Generate online augmentation
    """
    python online_augmentation/main.py --input data/output/gazette_sample_chunked.json --search

    """
    os.chdir('online_augmentation')
    extraction_command = 'python main.py --input data/input/ --output data/output -p -tp -ip'
    run_command(extraction_command)


if __name__ == "__main__":
    main()