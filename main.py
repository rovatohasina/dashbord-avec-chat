import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from datetime import datetime
from embedding.query_embeddings import query_pinecone
from data.wbdata_loader import get_live_wbdata

st.set_page_config(page_title="Prévision du PIB", layout="wide", initial_sidebar_state="expanded")

load_dotenv()


def create_prompt(question, wbdata_context, pdf_context):
    """
    Crée un prompt structuré pour répondre à une question sur un indicateur économique spécifique
    en n'utilisant que les données disponibles pour cet indicateur et cette année avec les données économiques (wbdata) et les explications (pdf),
    sans utiliser de numérotation dans la réponse.
    """
    wbdata_part = wbdata_context.strip()
    pdf_part = pdf_context.strip()

    if not wbdata_part and not pdf_part:
        return f"""
Tu es un assistant économique. Voici la question de l'utilisateur :

{question}

Réponds de manière fluide, claire et structurée. Si aucune information n’est disponible, réponds simplement :
"Je ne dispose pas de cette information."
"""
    else:
        prompt = f"""
Tu es un assistant économique intelligent. Réponds à la question suivante avec clarté et structure.

**Question :** {question}

**Instructions :**
- Si la question est juste une salutation (ex : bonjour, salut, hello), réponds uniquement par une salutation naturelle suivie d'une proposition d'aide,sans aucune donnée économique.
- Sinon,Réponds uniquement à **l'indicateur économique demandé** dans la question.
- Ne mentionne **aucun autre indicateur**, même si des données sont disponibles.
- Ne parle que de l’année explicitement demandée. Ignore les autres années.
- Si l’information est absente, écris : "Je ne dispose pas de cette information."
{wbdata_part}
- Ensuite, complète avec des explications tirées uniquement de ce contexte :
{pdf_part}

- Ne devine rien : si une information n'est pas présente, indique-le simplement ("Donnée non disponible").
- Ne répète pas la question. Concentre-toi sur une réponse directe, fluide et professionnelle.
"""
        return prompt


def ask_gemini(prompt):
    """
    Envoie le prompt à Gemini et retourne la réponse générée avec une bonne structure de phrase.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response

def save_conversation(question, response, file_path="logs/conversation_log.json"):
    if not os.path.exists("logs"):
        os.makedirs("logs")

    response_text = response.content if hasattr(response, "content") else str(response)

    conversation = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "response": response_text
    }

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(conversation)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    st.markdown("""
        <style>
            .chat-bubble-user {
                background-color: black;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 5px;
                text-align: right;
            }
            .chat-bubble-bot {
                background-color: black;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 5px;
                text-align: left;
            }
            .title {
                font-size: 32px;
                font-weight: bold;
                color: #004080;
            }
        </style>
    """, unsafe_allow_html=True)

# Initialisation de l'historique
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Formulaire de saisie 
df_pivot,filtered_data,forecast_df_PIB, filtered_df_depenses, filtered_df_recettes, filtered_ventillation,df_sunburst,selected_Années,selected_Année_ventillation,selected_Année,selected_Année_depenses,selected_Année_recettes=get_live_wbdata()
with st.sidebar.form("chat_form"):
    user_question = st.text_input("Votre question :", key="user_input")
    submitted = st.form_submit_button("Envoyer")

if submitted and (user_question or selected_Années or selected_Année_ventillation or selected_Année or selected_Année_depenses or selected_Année_recettes):
    with st.spinner("Analyse en cours..."):
        wbdata_context, pdf_context = query_pinecone(user_question,selected_Années,selected_Année_ventillation,selected_Année,selected_Année_depenses,selected_Année_recettes)
        prompt = create_prompt(user_question, wbdata_context, pdf_context)
        try:
            response = ask_gemini(prompt)
            response_text = response.content
        except Exception as e:
            response_text = f"Erreur : {e}"
    # Ajout des historiques de discussion
    st.session_state.chat_history.append(("Vous", user_question))
    st.session_state.chat_history.append(("Chat_economique", response_text))
    # Sauvegarde de la conversation
    save_conversation(user_question, response)
    st.session_state.submit = False
if "chat_history" in st.session_state and st.session_state.chat_history:
    for sender, msg in st.session_state.chat_history:
        bubble_class = "chat-bubble-user" if sender == "Vous" else "chat-bubble-bot"
        st.sidebar.markdown(f'<div class="{bubble_class}"><b>{sender} :</b><br>{msg}</div>', unsafe_allow_html=True)
else:
    st.sidebar.info("Aucune conversation pour le moment")
if __name__ == "__main__":
    main()


