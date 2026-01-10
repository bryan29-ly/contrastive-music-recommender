import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

from rhythmic_pattern_retrieval.models.encoder import RhythmicEncoder
from rhythmic_pattern_retrieval.utils.device import get_device


class GrooveMatcher:
    def __init__(self, model_path, db_path=None):
        self.device = get_device()
        print(f"Loading mmodel on {self.device}...")

        # Load the architecture
        self.model = RhythmicEncoder(projection_dim=128).to(self.device)

        # Load the weights
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Load the database
        self.database = None
        if db_path and Path(db_path).exists():
            print(f"Loading database from {db_path}...")
            self.database = pd.read_pickle(db_path)
            self.vector_matrix = np.stack(self.database['vector'].values)

    def get_embeddings(self, pt_file_path):
        """Transform a file .pt into a vector"""
        # 1. On charge l'objet complet (dictionnaire ou tenseur)
        data = torch.load(pt_file_path, weights_only=False)

        # 2. On vérifie le type pour extraire le bon tenseur
        if isinstance(data, dict):
            # Si c'est le nouveau format pré-traité (dict avec 'melspec' et 'valid_indices')
            spec = data['mel']
            # Note: On ignore 'valid_indices' pour l'instant et on fait confiance
            # à ta stratégie 'Peak Energy' ci-dessous qui est très robuste.
        else:
            # Si c'est un ancien fichier (juste le tenseur)
            spec = data

        # 3. On envoie sur le Device (GPU/MPS) maintenant qu'on a le tenseur
        spec = spec.to(self.device)

        # --- A PARTIR D'ICI, C'EST TON CODE D'ORIGINE ---

        # Peak energy strategy
        # 6 sec
        WINDOW = 512
        STRIDE = 300

        # Sécurité si le fichier est plus court que la fenêtre
        if spec.shape[-1] < WINDOW:
            pad = WINDOW - spec.shape[-1]
            spec = F.pad(spec, (0, pad))

        windows = spec.unfold(-1, WINDOW, STRIDE).permute(2, 0, 1, 3)

        # Keep the 50% more energic
        energies = windows.pow(2).mean(dim=(1, 2, 3))
        threshold = torch.quantile(energies, 0.5)
        active_windows = windows[energies >= threshold]

        if len(active_windows) == 0:
            active_windows = windows

        # Through the model
        with torch.no_grad():
            h, _ = self.model(active_windows)

        # Normalisation
        embedding = torch.mean(h, dim=0)
        embedding = F.normalize(embedding, p=2, dim=0)

        return embedding.cpu().numpy()

    def search(self, query_vector, top_k=5):
        """Search the neighbors in the loaded database."""
        if self.database is None:
            raise ValueError("No database loaded.")
        # Cosine distance
        scores = np.dot(self.vector_matrix, query_vector)
        # Sort the indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = self.database.iloc[top_indices].copy()
        results['score'] = scores[top_indices]
        return results
