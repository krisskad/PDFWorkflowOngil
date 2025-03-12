import os
import subprocess
import json
import pandas as pd
from es_load.main import upload_main

def read_json(file_path):
    # Open and read the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data into a Python dictionary

    return data

def add_search_to_chunks(chunk_data:dict, additional_data:dict, field_name = 'entity_search_results'):
    search_entities = {k:pd.DataFrame(v) for k,v in additional_data['online_search']['searched_entities'].items()}
    search_entitiy_df = pd.concat(search_entities, names = ['entity']).reset_index('entity')
    chunk_ids = search_entitiy_df['chunk_id'].unique()
    for i, chunk in enumerate(chunk_data['chunks']):
        if chunk['id'] in chunk_ids:
            chunk[field_name] = additional_data['online_search']['results']

    return chunk_data

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
        print(f"Chunking: {pdf_file}")
        prefix = os.path.splitext(pdf_file)[0]
        chunking_command = f'python -m chunking_cell.main --prefix "{prefix}" --document-id "{prefix}"'
        run_command(chunking_command)

        # Step 6: Generate online augmentation
        chunk_file = os.path.basename(pdf_file).replace(".pdf", "_chunked.json")
        chunk_file_path = f"data/output/{chunk_file}"
        print(f"online_augmentation: {chunk_file}")
        augmentation_command = f'python -m online_augmentation.main --input {chunk_file_path} --search'
        run_command(augmentation_command)

        # Step 7: merge chunked and augmented data
        chunk_augmented_file = os.path.basename(pdf_file).replace(".pdf", "_chunked_searched.json")
        chunk_augmented_file_path = f"data/output/augmented/{chunk_augmented_file}"
        chunked_data = read_json(chunk_file_path)
        chunked_data_augmented = read_json(chunk_augmented_file_path)
        final_data = add_search_to_chunks(chunk_data=chunked_data, additional_data=chunked_data_augmented)

        # Step 8: ES load
        if "chunks" in final_data:
            print("ES Loading")
            chunks = final_data["chunks"]
            upload_main(records=chunks, index_name="law-demo")


        # Step 9: case mine
        # starting casemine extraction
        casemine_command = f'python -m casemine_extraction.main'
        run_command(casemine_command)

        # es load casemine data
        case_details_json_path = f"data/output/case_details.json"
        case_details_json = read_json(case_details_json_path)
        upload_main(records=case_details_json, index_name="law-demo-casemine")


if __name__ == "__main__":
    main()