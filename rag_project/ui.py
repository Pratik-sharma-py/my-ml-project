import streamlit as st
from rag import query

st.set_page_config(page_title="RAG System", page_icon="🔍")

st.title("📚 Local RAG System")
st.markdown("Ask questions about your documents")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Your question:", placeholder="What is statistics?")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        result = query(question)

    st.session_state.history.append({
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"]
    })

if st.session_state.history:
    for item in reversed(st.session_state.history):
        st.markdown("---")
        st.markdown(f"**Question:** {item['question']}")
        st.markdown(f"**Answer:** {item['answer']}")
        st.markdown(f"**Sources:** {', '.join(set(item['sources']))}")