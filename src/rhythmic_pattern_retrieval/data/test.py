# debug_paths.py
from rhythmic_pattern_retrieval.config import RAW_DATA_DIR
import os

print(f"📍 Je cherche dans : {RAW_DATA_DIR}")
print(f"📂 Ce dossier existe ? : {RAW_DATA_DIR.exists()}")

# On regarde ce qu'il y a dedans
print("\ncontents de raw :")
try:
    print(list(RAW_DATA_DIR.iterdir()))
except Exception as e:
    print(e)

# On teste la recherche de MP3
mp3s = list(RAW_DATA_DIR.glob("**/*.mp3"))
print(f"\n🎵 Nombre de MP3 trouvés : {len(mp3s)}")

if len(mp3s) > 0:
    print(f"   Exemple : {mp3s[0]}")
else:
    print("❌ PROBLÈME : Je ne trouve aucun MP3. Le dossier est-il bien dézippé ?")
