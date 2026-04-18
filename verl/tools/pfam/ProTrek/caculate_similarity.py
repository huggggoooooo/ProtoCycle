import torch

from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
from utils.foldseek_util import get_struc_seq

# Load model
config = {
    "protein_config": "weights/ProTrek_650M/esm2_t33_650M_UR50D",
    "text_config": "weights/ProTrek_650M/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "structure_config": "weights/ProTrek_650M/foldseek_t30_150M",
    "from_checkpoint": "weights/ProTrek_650M/ProTrek_650M.pt"
}

device = "cuda"
model = ProTrekTrimodalModel(**config).eval().to(device)

# Load protein and text
pdb_path = "example/8ac8.cif"
seqs = get_struc_seq("bin/foldseek", pdb_path, ["A"])["A"]
aa_seq = seqs[0]
foldseek_seq = seqs[1].lower()
text = "Replication initiator in the monomeric form, and autogenous repressor in the dimeric form."

with torch.no_grad():
    # Obtain protein sequence embedding
    seq_embedding = model.get_protein_repr([aa_seq])
    print("Protein sequence embedding shape:", seq_embedding.shape)
    
    # Obtain protein structure embedding
    struc_embedding = model.get_structure_repr([foldseek_seq])
    print("Protein structure embedding shape:", struc_embedding.shape)
    
    # Obtain text embedding
    text_embedding = model.get_text_repr([text])
    print("Text embedding shape:", text_embedding.shape)
    
    # Calculate similarity score between protein sequence and structure
    seq_struc_score = seq_embedding @ struc_embedding.T / model.temperature
    print("Similarity score between protein sequence and structure:", seq_struc_score.item())

    # Calculate similarity score between protein sequence and text
    seq_text_score = seq_embedding @ text_embedding.T / model.temperature
    print("Similarity score between protein sequence and text:", seq_text_score.item())
    
    # Calculate similarity score between protein structure and text
    struc_text_score = struc_embedding @ text_embedding.T / model.temperature
    print("Similarity score between protein structure and text:", struc_text_score.item())
   

"""
Protein sequence embedding shape: torch.Size([1, 1024])
Protein structure embedding shape: torch.Size([1, 1024])
Text embedding shape: torch.Size([1, 1024])
Similarity score between protein sequence and structure: 28.506675720214844
Similarity score between protein sequence and text: 17.842409133911133
Similarity score between protein structure and text: 11.866174697875977
"""