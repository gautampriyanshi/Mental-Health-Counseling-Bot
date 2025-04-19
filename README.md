Overview

This project is a Mental Health Counselling Assistant Bot built using Gradio and LangChain. It allows users to interact with the chatbot via text or voice input to receive guidance on mental health topics.



Features

Chatbot Interface: Users can ask mental health-related questions.

Speech Recognition: Converts spoken input into text for interaction.

LangChain Processing: Uses Hugging Face models for responses.

Gradio UI: Provides an easy-to-use interface for users.



Set up Hugging Face API Token

Replace your_new_huggingface_token in the code with your Hugging Face API Token.

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_new_huggingface_token"



How It Works

Users can type or speak their questions.

The bot processes text input or converts speech to text using Google Speech Recognition.

It generates responses using LangChain and Hugging Face models.

The conversation is displayed in a chatbot format.



