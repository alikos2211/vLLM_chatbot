from pinecone import Pinecone
from pinecone import ServerlessSpec
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
index_name = os.getenv("PINECONE_INDEX")


if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
