## using elastic 8.12
import os
from elasticsearch import Elasticsearch

_ES_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")

def get_client():
    return Elasticsearch(_ES_URL, request_timeout=30)