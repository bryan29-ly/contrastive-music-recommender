
import os
import certifi
from rhythmic_pattern_retrieval.data.preprocessor import Preprocessor

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

if __name__ == "__main__":
    print("Preprocessing with Preprocessor class...")
    preprocessor = Preprocessor(limit=5)
    preprocessor.preprocess_dataset()
