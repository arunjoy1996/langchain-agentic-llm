import streamlit as st
import requests
import re

st.set_page_config(page_title="LLM Agent Stream", layout="wide")
st.title("🧠 Chain-of-Thought Agent with Streaming and Image Generation")

# Initialize session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask me anything 👇", placeholder="e.g., What is 23 multiplied by 9 plus 6?", key="input")

if st.button("Submit") and query:
    
    

    response = requests.get(
        "http://localhost:8001/stream",
        params={"query": query},
        stream=True,
    )
    st.write("### 💬 Response:")
    response_area = st.empty()
    # Display the full chat history
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            st.markdown(f"**🧑 You:** {entry['content']}")
        else:
            st.markdown(f"**🤖 AI:** {entry['content']}")
            for img_url in entry.get("images", []):
                st.image(img_url)
    full_response = ""

    image_urls = []

    for line in response.iter_lines():
        if line:
            decoded = line.decode("utf-8").replace("data: ", "").strip()
            if decoded.endswith(".png") or ".png" in decoded:
                parts = decoded.split(" ")
                for part in parts:
                    if part.startswith("http"):
                        image_urls.append(part)
                        response_area.image(part)
            else:                

                # Append without extra space
                full_response += " " + decoded

                response_area.markdown(f"**🤖 AI:** {full_response}")


    # Append response
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "ai", "content": full_response, "images": image_urls})

