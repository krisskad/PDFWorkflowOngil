from langchain_openai import OpenAIEmbeddings
from elasticsearch import Elasticsearch, helpers
import os
import json
import logging
import pandas as pd
from dotenv import load_dotenv


if os.path.exists(".env"):
    load_dotenv()
# Then try to load from the pdf_extraction_cell directory
elif os.path.exists("es_load/.env"):
    load_dotenv("es_load/.env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('es_log.log', mode='w')
    ]
)

logger = logging.getLogger("es_load")


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ES_HOST = os.environ.get("ES_HOST")
ES_USERNAME = os.environ.get("ES_USERNAME")
ES_PASSWORD = os.environ.get("ES_PASSWORD")


chunk_mapping = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "type": {"type": "keyword"},
            "page_number": {"type": "integer"},
            "content": {"type": "text"},
            "file_id": {"type": "keyword"},
            "page_title": {"type": "text"},
            "dynamic_fields": {"type": "object"},
            "embedding": {
                "type": "dense_vector",
                "dims": 1536
            }
        }
    }
}


# Initialize OpenAI embedding model (Ada)
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)


def get_es_client():
    return Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USERNAME, ES_PASSWORD),
        verify_certs=False
    )

def read_json(file_path):
    # Open and read the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load JSON data into a Python dictionary

    return data

def create_es_index(index_name):
    es = get_es_client()

    # Delete index if it exists (optional)
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    # Create index
    es.indices.create(index=index_name, body=chunk_mapping)
    return f"Index {index_name} created successfully!"


def add_search_to_chunks(chunk_data:dict, additional_data:dict, field_name = 'entity_search_results'):
    search_entities = {k:pd.DataFrame(v) for k,v in additional_data['online_search']['searched_entities'].items()}
    search_entitiy_df = pd.concat(search_entities, names = ['entity']).reset_index('entity')
    chunk_ids = search_entitiy_df['chunk_id'].unique()
    for i, chunk in enumerate(chunk_data['chunks']):
        if chunk['id'] in chunk_ids:
            chunk[field_name] = additional_data['online_search']['results']


def upload_records_to_elasticsearch(records, index_name):
    es = get_es_client()

    # Check if the index exists
    create_es_index(index_name=index_name)

    left_docs = []
    for record in records:
        doc_id = record["id"]
        doc_type = record["type"]

        # Separate fixed and dynamic fields
        fixed_fields = {}
        dynamic_fields = {}

        for key, value in record.items():
            if key in chunk_mapping["mappings"]["properties"]:
                fixed_fields[key] = value
            else:
                dynamic_fields[key] = value

        # Add dynamic fields to a "dynamic_fields" object
        fixed_fields["dynamic_fields"] = dynamic_fields

        # Generate vector embedding for the record
        fixed_fields["embedding"] = embedding_model.embed_query(str(record))

        # Upload the document to Elasticsearch (explicitly specify the index)
        try:
          es.index(index=index_name, id=doc_id, body=record)
        except Exception as e:
          print(f"Error uploading record with ID {record['id']}: {e}")
          left_docs.append(record)
          continue

    return left_docs


def get_index_stats(index_name):
    es = get_es_client()
    if not es.indices.exists(index=index_name):
        return "Index does not exist"

    stats = es.count(index=index_name)
    return {"index": index_name, "document_count": stats["count"]}

def semantic_search(index_name, query_text, top_k=5):
    es = get_es_client()
    query_embedding = embedding_model.embed_query(query_text)

    search_body = {
        "size": top_k,
        "_source": ["text"],
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        }
    }

    results = es.search(index=index_name, body=search_body)
    return [{"text": hit["_source"]["text"], "score": hit["_score"]} for hit in results["hits"]["hits"]]


def upload_main(records: list):
    index_name = "law-demo"
    upload_records_to_elasticsearch(
        records=records,
        index_name=index_name
    )

    stat = get_index_stats(index_name=index_name)
    print(stat)