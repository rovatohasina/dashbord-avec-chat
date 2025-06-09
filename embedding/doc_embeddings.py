import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from data.wbdata_loader import get_live_wbdata
import pandas as pd
# Charger les variables d'environnement
load_dotenv()

# Charger les fichiers PDF depuis un dossier
def get_pdf_documents(directory="data/doc"):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, filename))
            pages = loader.load()
            docs.extend(pages)
    return docs
def get_wbdata_documents():
    df_pivot,filtered_data,forecast_df_PIB, filtered_df_depenses, filtered_df_recettes, filtered_ventillation,df_sunburst_all= get_live_wbdata()

    dfs = [
        ("actuel",filtered_data),
        ("Prévision", forecast_df_PIB),
        ("tableau", df_pivot),
        ("Dépenses", filtered_df_depenses),
        ("Recettes", filtered_df_recettes),
        ("Ventillation", filtered_ventillation),
        ("RepartitionVentillation", df_sunburst_all)
    ]
    # Ajoute une colonne "type" dans chaque DataFrame
    for nom_df, df in dfs:
        df['type'] = nom_df.lower()

    # Concatène tous les DataFrames en un seul
    df_wbdata = pd.concat([df for _, df in dfs], ignore_index=True)
    wbdata_docs = []

    for nom_df, df in dfs:
        for _, row in df.iterrows():
            try:
                année = int(row['Année']) if 'Année' in row else "Inconnue"
            except:
                année = "Inconnue"

            # parts = [f"[{nom_df}] En {année} :"]
            doc_type = row['type']  # récupère le type ici

            parts = [f"[{nom_df}] En {année} :"]

            for col in df.columns:
                if col != 'Année' and pd.notnull(row[col]):
                    try:
                        value = round(row[col], 2)
                        parts.append(f"{col} était de {value}")
                    except:
                        parts.append(f"{col} : {row[col]}")

            text = ". ".join(parts) + "."
            wbdata_docs.append(
                {"Année": année,
                 "type":doc_type,
                 "text": text}
                )
    return wbdata_docs


# Embedding des documents
def embed_pdfs_and_wbdata_and_check(df, pdf_documents):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # WBData
    wbdata_texts = df['text'].tolist()
    wbdata_embeddings = embed_model.embed_documents(wbdata_texts)
    wb_vectors = [(text, vec, "wbdata") for text, vec in zip(wbdata_texts, wbdata_embeddings)]

    # PDF
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splitted_docs = splitter.split_documents(pdf_documents)
    pdf_texts = [doc.page_content for doc in splitted_docs]
    pdf_embeddings = embed_model.embed_documents(pdf_texts)
    pdf_vectors = [(text, vec, "pdf") for text, vec in zip(pdf_texts, pdf_embeddings)]

    # 5. Affichage pour vérification
    for i, emb in enumerate(pdf_embeddings[:3]):
        print(f"PDF Embedding {i} : {len(emb)} dimensions")

    for i, emb in enumerate(wbdata_embeddings[:3]):
        print(f"WBData Embedding {i} : {len(emb)} dimensions")

    # Combiner les deux
    return wb_vectors + pdf_vectors

