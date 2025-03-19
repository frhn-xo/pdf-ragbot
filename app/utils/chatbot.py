import streamlit as st
from collections import defaultdict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

def get_context_retriever_chain(vectordb):
    """
    Create a context retriever chain for generating responses based on the chat history and vector database

    Parameters:
    - vectordb: Vector database used for context retrieval

    Returns:
    - retrieval_chain: Context retriever chain for generating responses
    """
    # Load environment variables (gets api keys for the models)
    load_dotenv()
    
    # Initialize the model with the correct model name
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro",
        temperature=0.1,  # Lower temperature for more deterministic outputs
        convert_system_message_to_human=True
    )
    
    # Create a retriever with a small k value to limit results to most relevant
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a document Q&A assistant that ONLY answers questions based on the provided document context.

IMPORTANT RULES:
1. If the context does not contain information directly relevant to the user's question, say: "I don't have information about that in the uploaded documents. Please ask something related to the documents I have access to."
2. NEVER use your general knowledge to answer questions - rely EXCLUSIVELY on the provided context.
3. NEVER make up or hallucinate information if the context doesn't contain a clear answer.
4. Don't mention the concept of context in your answers - just answer as if you're referring to the documents.
5. Keep your answers concise and directly related to what's in the documents.

Here is the document context to use for answering:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # Create chain for generating responses and a retrieval chain
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, chain)
    return retrieval_chain

def manually_check_response(response, query):
    """
    Perform additional validation on the response to enforce document-only answers
    
    Parameters:
    - response (str): The generated response
    - query (str): The user query
    
    Returns:
    - valid_response (str): Either the original response or a standard refusal message
    """
    # List of common greetings and non-document queries
    greeting_patterns = ["hello", "hi ", "hey", "greetings", "what's up", "howdy"]
    
    # Direct answer patterns that often indicate the model is using general knowledge
    generic_answer_patterns = ["I don't have specific information", "Based on my knowledge", 
                             "In general", "Typically", "Usually", "As a language model"]
    
    # Check if query is just a greeting
    if any(query.lower().startswith(pattern) for pattern in greeting_patterns) and len(query.split()) < 3:
        return "I'm a document assistant. Please ask me questions about the documents you've uploaded."
    
    # Check if answer looks like it's from general knowledge
    if any(pattern in response for pattern in generic_answer_patterns):
        return "I don't have information about that in the uploaded documents. Please ask something related to the documents I have access to."
    
    # Check if response is too short and not a refusal (might be a simple factoid from general knowledge)
    if len(response.split()) < 15 and not "don't have information" in response.lower():
        return "I don't have information about that in the uploaded documents. Please ask something related to the documents I have access to."
        
    return response

def get_response(question, chat_history, vectordb):
    """
    Generate a response to the user's question based on the chat history and vector database

    Parameters:
    - question (str): The user's question
    - chat_history (list): List of previous chat messages
    - vectordb: Vector database used for context retrieval

    Returns:
    - response: The generated response
    - context: The context associated with the response
    """
    try:
        chain = get_context_retriever_chain(vectordb)
        chain_response = chain.invoke({"input": question, "chat_history": chat_history})
        
        # Extract response and context
        response = chain_response["answer"]
        context = chain_response.get("context", [])
        
        # Perform additional validation on the response
        validated_response = manually_check_response(response, question)
        
        return validated_response, context
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        # Return a fallback response when an error occurs
        return f"I'm having trouble processing your request. Error: {str(e)}", []

def chat(chat_history, vectordb):
    """
    Handle the chat functionality of the application

    Parameters:
    - chat_history (list): List of previous chat messages
    - vectordb: Vector database used for context retrieval

    Returns:
    - chat_history: Updated chat history
    """
    user_query = st.chat_input("Ask a question about your documents:")
    if user_query is not None and user_query != "":
        try:
            # Generate response based on user's query, chat history and vectorstore
            response, context = get_response(user_query, chat_history, vectordb)
            
            # Update chat history
            chat_history = chat_history + [HumanMessage(content=user_query), AIMessage(content=response)]
            
            # Display source of the response on sidebar only if it's not a refusal message
            with st.sidebar:
                if context and not "don't have information" in response.lower():
                    st.subheader("Sources:")
                    metadata_dict = defaultdict(list)
                    for metadata in [doc.metadata for doc in context]:
                        metadata_dict[metadata['source']].append(metadata['page'])
                    for source, pages in metadata_dict.items():
                        st.write(f"Source: {source}")
                        st.write(f"Pages: {', '.join(map(str, pages))}")
                elif "don't have information" in response.lower():
                    st.info("No relevant information found in documents for this query.")
        except Exception as e:
            st.error(f"Error in chat processing: {str(e)}")
            chat_history.append(HumanMessage(content=user_query))
            chat_history.append(AIMessage(content=f"I encountered an error processing your request. Please try again or check the system logs for more information. Error: {str(e)}"))
    
    # Display chat history
    for message in chat_history:
        with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
            st.write(message.content)
    
    return chat_history