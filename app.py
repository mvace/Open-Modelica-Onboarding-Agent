import streamlit as st
from settings import settings
from tools.doc_search_tool import _search_docs
from chains.answer_chain import answer_with_citations

st.set_page_config(page_title="OpenModelica Onboarding Assistant", layout="wide")
st.title("OpenModelica Onboarding Assistant")
st.caption(
    "Ask questions about the OpenModelica User’s Guide. Answers are grounded in the PDF with page citations."
)
# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# Chat input
q = st.chat_input("Ask about OpenModelica…")
if q:
    # Show user message
    st.session_state.messages.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documentation and generating answer..."):
            passages = _search_docs(q, k=5)
            answer = answer_with_citations(q, passages)
            st.markdown(answer)
        st.session_state.messages.append(("assistant", answer))
