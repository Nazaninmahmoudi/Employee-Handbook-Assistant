import os
import uuid
import pickle
import tempfile

import gradio as gr
import soundfile as sf

from dotenv import load_dotenv

from kokoro import KPipeline

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from hybrid_retriever import HybridRRF



load_dotenv()



K = 5



bge_embeddings = HuggingFaceEmbeddings(

    model_name="BAAI/bge-base-en-v1.5",

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    }

)



bge_vector_store = Chroma(

    collection_name="final_rag_collection",

    persist_directory="./chroma_db",

    embedding_function=bge_embeddings

)



bge_retriever = bge_vector_store.as_retriever(

    search_kwargs={
        "k": K
    }

)



with open(

    "./bm25_retriever.pkl",

    "rb"

) as f:

    bm25_retriever = pickle.load(f)



hybrid_retriever = HybridRRF(

    retrievers=[

        bge_retriever,

        bm25_retriever

    ],

    weights=[

        0.7,

        0.3

    ],

    top_k=K

)



llm = ChatGroq(

    model="llama-3.3-70b-versatile",

    temperature=0,

    streaming=False,

    api_key=os.getenv(
        "GROQ_API_KEY"
    )

)



store = {}



def get_session_history(session_id):

    if session_id not in store:

        store[session_id] = InMemoryChatMessageHistory()


    return store[session_id]



def get_session_id():

    return str(uuid.uuid4())




memory_prompt = ChatPromptTemplate.from_messages(

    [

        (

            "system",

            """
You are a helpful assistant.

Answer only using the provided context.

Context:
{context}
"""

        ),

        MessagesPlaceholder(

            variable_name="history"

        ),

        (

            "human",

            "{question}"

        )

    ]

)



def format_docs(docs):

    return "\n\n".join(

        doc.page_content

        for doc in docs

    )




def retrieve_context(inputs):

    question = inputs["question"]

    docs = hybrid_retriever.invoke(question)

    context = format_docs(docs)


    return {

        "context": context,

        "question": question,

        "history": inputs.get(

            "history",

            []

        )

    }




rag_chain = (

    retrieve_context

    |

    memory_prompt

    |

    llm

)



rag_chatbot = RunnableWithMessageHistory(

    rag_chain,

    get_session_history,

    input_messages_key="question",

    history_messages_key="history"

)




def chat_response(

    message,

    history,

    session_id

):

    if history is None:

        history = []


    if session_id is None:

        session_id = get_session_id()


    if message is None or message.strip() == "":

        return (

            history,

            session_id,

            "",

            ""

        )


    response = rag_chatbot.invoke(

        {
            "question": message
        },

        config={

            "configurable": {

                "session_id": session_id

            }

        }

    )


    answer = response.content


    history.append(

        {

            "role": "user",

            "content": message

        }

    )


    history.append(

        {

            "role": "assistant",

            "content": answer

        }

    )


    return (

        history,

        session_id,

        "",

        answer

    )




tts_pipeline = KPipeline(

    lang_code="a"

)



def text_to_speech(text):

    if text is None or text.strip() == "":

        return None


    output_file = tempfile.NamedTemporaryFile(

        suffix=".wav",

        delete=False

    ).name



    generator = tts_pipeline(

        text,

        voice="af_heart"

    )


    audio = None


    for _, _, wav in generator:

        audio = wav



    sf.write(

        output_file,

        audio,

        24000

    )


    return output_file




def clear_chat(session_id):

    if session_id in store:

        del store[session_id]


    new_session = get_session_id()


    return (

        [],

        new_session,

        "",

        "",

        None

    )




with gr.Blocks() as demo:


    gr.Markdown(

        """
# 🤖 Employee Handbook Assistant

Ask questions about company policies and employee information.

Powered by Hybrid RAG + LLM + Memory + TTS
"""

    )



    chatbot = gr.Chatbot(

        height=500,

        type="messages"

    )



    session_state = gr.State(None)

    answer_state = gr.State("")



    with gr.Row():

        message_box = gr.Textbox(

            label="Question",

            placeholder="Ask about employee handbook..."

        )


        send_button = gr.Button(

            "Send"

        )



    with gr.Row():

        speak_button = gr.Button(

            "Read Answer"

        )


        new_chat = gr.Button(

            "New Chat"

        )



    voice_output = gr.Audio(

        label="Answer Voice"

    )



    send_button.click(

        chat_response,

        inputs=[

            message_box,

            chatbot,

            session_state

        ],

        outputs=[

            chatbot,

            session_state,

            message_box,

            answer_state

        ]

    )



    message_box.submit(

        chat_response,

        inputs=[

            message_box,

            chatbot,

            session_state

        ],

        outputs=[

            chatbot,

            session_state,

            message_box,

            answer_state

        ]

    )



    speak_button.click(

        text_to_speech,

        inputs=[

            answer_state

        ],

        outputs=[

            voice_output

        ]

    )



    new_chat.click(

        clear_chat,

        inputs=[

            session_state

        ],

        outputs=[

            chatbot,

            session_state,

            message_box,

            answer_state,

            voice_output

        ]

    )




demo.launch(

    server_name="127.0.0.1",

    server_port=7848,

    share=False

)