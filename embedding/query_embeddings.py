import os
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Initialiser Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

# Définir l'index Pinecone
index_name = "nouvel"
index = pc.Index(index_name)

# Initialiser le modèle d'embedding
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def vectorize_question(question: str):
    return embed_model.embed_query(question)

def query_pinecone(question: str,
                   selected_Années=None,
                   selected_Année_ventillation=None,
                   selected_Année=None,
                   selected_decade_depenses=None,
                   selected_decade_recettes=None,
                   top_k=None):
    # Vectorisation question
    question_embedding = vectorize_question(question)

    filtres = []
    total_annees = 0

    if selected_Années:
        filtres.append({
            "$and": [
                {"Année": {"$gte": selected_Années[0], "$lte": selected_Années[1]}},
                {"type": "actuel"}
            ]
        })
        total_annees += selected_Années[1] - selected_Années[0] + 1

    if selected_Année_ventillation:
        filtres.append({
            "$and": [
                {"Année": {"$gte": selected_Année_ventillation[0], "$lte": selected_Année_ventillation[1]}},
                {"type": "ventillation"}  
            ]
        })
        total_annees += selected_Année_ventillation[1] - selected_Année_ventillation[0] + 1

    if selected_Année:
        filtres.append({
            "$and": [
                {"Année": int(selected_Année)},
                {"type": "repartitionventillation"}
            ]
        })
        total_annees += 1

    if selected_decade_depenses:
        filtres.append({
            "$and": [
                {"Année": int(selected_decade_depenses)},
                {"type": "dépenses"}
            ]
        })
        total_annees += 1

    if selected_decade_recettes:
        filtres.append({
            "$and": [
                {"Année": int(selected_decade_recettes)},
                {"type": "recettes"}
            ]
        })
        total_annees += 1
    top_k = total_annees if total_annees > 0 else 20
    pinecone_filter = {
    "$and": [
        {"source": "wbdata"},
        {"$or": filtres} 
    ]
}
    # Requête vers les données WBData 
    wbdata_results = index.query(
        vector=question_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
        filter=pinecone_filter
    )
    # Requête vers les documents PDF 
    pdf_results = index.query(
        vector=question_embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
        filter={"source": {"$eq": "pdf"}}
    )
            # Résultats WBData 
    wbdata_contexts = []
    if wbdata_results and wbdata_results.get("matches"):
        for match in wbdata_results.get("matches", []):
            texte = match.get("metadata", {}).get("text", "").strip()
            if texte:
                wbdata_contexts.append(f"WBData : {texte}")
    else:
        print("Recherche WBData NON OK - Aucun résultat trouvé.")

            # Résultats PDF 
    pdf_contexts = []
    if pdf_results and pdf_results.get("matches"):
        for match in pdf_results.get("matches", []):
            texte = match.get("metadata", {}).get("text", "").strip()
            if texte:
                pdf_contexts.append(f"PDF: {texte}")
    else:
        pdf_contexts.append("Aucun contexte explicatif PDF trouvé.")
        
            # Fusion
    wbdata_context = ("\n".join(wbdata_contexts))
    pdf_context = ("\n".join(pdf_contexts))

    return wbdata_context,pdf_context


