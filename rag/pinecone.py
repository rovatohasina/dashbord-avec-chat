from dotenv import load_dotenv

import os
from pinecone import Pinecone

load_dotenv()

def init_pinecone():
    # Charger la clé API et l'environnement depuis les variables d'environnement
    api_key = os.getenv("PINECONE_API_KEY")
    
    if not api_key:
        raise ValueError("❌ Clé API non trouvée.")
    
    # Initialiser le client Pinecone avec la clé API
    pc = Pinecone(api_key=api_key)

    index_name = "nouvel"
    # Accéder à l'index créé ou existant
    index = pc.Index(index_name)
    
    return pc, index
    # # Vérifier si l'index existe déjà
    # index_names = [index.name for index in pc.list_indexes()]
    
    # # Si l'index n'existe pas, le créer avec ServerlessSpec correctement spécifié
    # if index_name not in index_names:
    #     pc.create_index(
    #         name=index_name,
    #         dimension=384,
    #         metric="cosine",
    #         spec=ServerlessSpec(cloud="aws", region=region)  # Correctement spécifié ici
    #     )



