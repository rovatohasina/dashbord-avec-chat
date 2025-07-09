from pinecone import Index
from rag.pinecone import init_pinecone
from embedding.doc_embeddings import embed_pdfs_and_wbdata_and_check, get_pdf_documents, get_wbdata_documents
import pandas as pd

def insert_vectors():
    # Charger les donnees
    pdf_docs = get_pdf_documents("data/doc")
    wbdata_docs = get_wbdata_documents()
    df_wbdata = pd.DataFrame(wbdata_docs)
    
    vectors = embed_pdfs_and_wbdata_and_check(df_wbdata, pdf_docs)
    pc, index = init_pinecone()
    inserted_ids = []
    
    # Insérer WBData avec annee
    for i in range(len(df_wbdata)):
        text, embedding, source = vectors[i]
        vector_id = f"{source}-{i}"
        annee = df_wbdata.loc[i, "Année"]
        doc_type = df_wbdata.loc[i, "type"]
        index.upsert([{
            "id": vector_id,
            "values": embedding,
            "metadata": {"source": source, "text": text, "Année": annee, "type": doc_type}
        }])
        inserted_ids.append(vector_id)
    
    # Insérer PDF 
    offset = len(df_wbdata)
    for i in range(offset, len(vectors)):
        text, embedding, source = vectors[i]
        vector_id = f"{source}-{i - offset}"
        index.upsert([{
            "id": vector_id,
            "values": embedding,
            "metadata": {"source": source, "text": text}
        }])
        inserted_ids.append(vector_id)
    
    print("Vecteurs insérés avec succès dans Pinecone.")
    for vid in inserted_ids:
        print(vid)

if __name__ == "__main__":
    insert_vectors()

