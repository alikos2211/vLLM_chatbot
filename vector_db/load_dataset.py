import os
os.environ["KAGGLEHUB_CACHE"] = "./VectorDB/data"
import kagglehub

path = kagglehub.dataset_download("ffatty/plain-text-wikipedia-simpleenglish")
