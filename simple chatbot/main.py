import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="Gemini Chatbot", page_icon="", layout="centered")

# Title
st.title("Chatbot with Memory")
st.markdown("Chat with Google's Gemini AI - Your conversation is remembered!")

# Get API key from environment
api_key = os.getenv("GOOGLE_API_KEY")

# # Sidebar
# with st.sidebar:
#     st.header("⚙️ Configuration")
    
#     # Show API key status
#     if api_key:
#         st.success("✅ API Key loaded from environment")
#     else:
#         st.error("❌ API Key not found in environment")
#         st.info("Please set GOOGLE_API_KEY in your .env file")
    
#     st.markdown("---")
#     st.markdown("### About")
#     st.info("This chatbot uses Google Gemini AI with conversation memory powered by LangChain.")
#     st.markdown("[Get API Key](https://makersuite.google.com/app/apikey)")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Function to initialize conversation
def initialize_conversation(api_key):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True,
            verbose=True
        )
        
        conversation = ConversationChain(
            llm=llm,
            memory=st.session_state.memory,
            verbose=False
        )
        return conversation
    except Exception as e:
        st.error(f"Error initializing conversation: {str(e)}")
        return None

# Function to reset chat
def reset_chat():
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.conversation = None
    st.success("Chat history cleared! Starting fresh conversation.")

# Reset button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Reset Chat", use_container_width=True, type="secondary"):
        reset_chat()
        st.rerun()

st.markdown("---")

if not api_key:
    st.warning("⚠️ Please enter your Google Gemini API Key in the sidebar to start chatting.")
    st.stop()

# Initialize conversation if not already done
if st.session_state.conversation is None:
    st.session_state.conversation = initialize_conversation(api_key)
    if st.session_state.conversation is None:
        st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.conversation.predict(input=prompt)
                st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "Powered by Google Gemini AI & LangChain"
    "</div>",
    unsafe_allow_html=True
)