from rhythmic_pattern_retrieval.models.encoder import RhythmicEncoder

model = RhythmicEncoder(projection_dim=128)

# Calculer les paramètres entraînables
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Trainable Parameters: {total_params:,}")
