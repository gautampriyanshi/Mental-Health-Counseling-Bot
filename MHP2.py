import os
import gradio as gr
import speech_recognition as sr
from huggingface_hub import login
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA

login("Your-huggingface-token")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "Your-huggingface-token"

chain = None

# PDF Processing
def process_pdf(file_paths):
    global chain
    if not file_paths:
        return "No files uploaded. Please upload PDF files."
    
    if not all(file_path.endswith('.pdf') for file_path in file_paths):
        return "Please upload valid PDF files."
    
    try:
        documents = []
        for file_path in file_paths:
            try:
                loader = PyMuPDFLoader(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
            except Exception as e:
                return f"Error loading {file_path}: {str(e)}"
        
        if not documents:
            return "No text found in the uploaded PDFs."
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_text = text_splitter.split_documents(documents)
        
        if not split_text:
            return "No text found after splitting the documents."
        
        db = Chroma.from_documents(split_text, embeddings)
        llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.3")
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
        
        return "Dataset processed successfully! You can now ask questions."
    except Exception as e:
        return f"Error processing PDFs: {str(e)}"

# QA Function
def answer_query(query):
    if chain is None:
        return "Please upload and process Dataset(Pdf's) first."
    
    try:
        result = chain.run(query)
        formatted_result = f"{query}\n {result.strip()}"
        return formatted_result
    except Exception as e:
        return f"Error during query: {str(e)}"

# Programmatic Upload
def programmatic_upload_and_process():
    pdf_paths = [
        r"D:\MHP\combinedpdf_dataset.pdf"
    ]
    
    try:
        status = process_pdf(pdf_paths)
        global chain
        if chain is None:
            raise Exception("Failed to initialize the QA chain.")
        return "Dataset processed successfully! You can now ask questions."
    except Exception as e:
        return f"Error: {str(e)}"

# Speech Recognition from Audio
def transcribe_audio(audio):
    if audio is None:
        return "No audio input provided."
    
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        return "Sorry, I couldn't understand the audio."

# Handle Text Interaction
def respond(message, chat_history):
    if chain is None:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "Dataset's are still being processed. Please wait..."})
        return chat_history

    response = answer_query(message)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})
    return chat_history



# Handle Audio Interaction
def handle_audio(audio_path, chat_history):
    transcribed = transcribe_audio(audio_path)
    chat_history.append({"role": "user", "content": transcribed})

    if transcribed.startswith("Sorry") or transcribed.startswith("No audio"):
        chat_history.append({"role": "assistant", "content": transcribed})
        return chat_history

    response = answer_query(transcribed)
    chat_history.append({"role": "assistant", "content": response})
    return chat_history




# Clear Chat
def clear_chat():
    return []


# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖🌿 Mental Health Counselling Bot
        Welcome to your personal **Mental Health Assistant**.
        Feel free to ask questions or speak them out — I'm here to help. 😊
        """,
        elem_id="title",
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="💬 Conversation", type="messages", height=450)
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### 🛠 Controls")
            clear = gr.Button("🧹 Clear Chat", variant="secondary")
            submit_audio = gr.Button("🎤 Submit Audio", variant="primary")

    with gr.Row():
        msg = gr.Textbox(
            label="📝 Type your question",
            placeholder="e.g., How can I manage anxiety?",
            lines=1,
            scale=3
        )

    with gr.Accordion("🎧 Or speak your question", open=False):
        audio_input = gr.Audio(type="filepath", label="Upload or Record Audio", show_label=True)

    def respond(message, chat_history):
        if chain is None:
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": "Dataset's are still being processed. Please wait..."})
            return chat_history
        response = answer_query(message)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": response})
        return chat_history

    # def respond(message, chat_history):
    #     if chain is None:
    #         chat_history.append((message, "Dataset's are still being processed. Please wait..."))
    #         return chat_history
    #     response = answer_query(message)
    #     chat_history.append((message, response))
    #     return chat_history
    
    def clear_chat():
        return []

    def on_load():
        status = programmatic_upload_and_process()
        return [{"role": "assistant", "content": status}]

    msg.submit(respond, [msg, chatbot], chatbot)
    submit_audio.click(handle_audio, [audio_input, chatbot], chatbot)
    clear.click(clear_chat, None, chatbot, queue=False)

    demo.load(on_load, None, chatbot)

demo.launch()
