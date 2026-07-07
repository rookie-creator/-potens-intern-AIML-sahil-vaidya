"""
ui.py
-----

Professional Streamlit UI for the RAG Document QA System.

Theme
-----
• Matte Black Background
• Glassmorphism Cards
• Gold Accent Color
• White Typography
• Smooth Hover Effects

Pages
-----
1. Ask Questions
2. Contradiction Checker
"""

import requests
import streamlit as st

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

API_URL = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------
# Custom CSS
# -------------------------------------------------------

st.markdown(
    """
<style>

/* ===========================
   Import Font
=========================== */

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}


/* ===========================
   Background
=========================== */

.stApp{

    background:
        radial-gradient(circle at top left,#222 0%,#111 35%,#050505 100%);

    color:white;
}


/* ===========================
   Sidebar
=========================== */

section[data-testid="stSidebar"]{

    background:rgba(20,20,20,.95);

    border-right:1px solid rgba(255,215,0,.25);

}


/* ===========================
   Glass Card
=========================== */

.glass{

    background:rgba(255,255,255,.06);

    border:1px solid rgba(255,215,0,.25);

    backdrop-filter:blur(16px);

    border-radius:22px;

    padding:25px;

    margin-bottom:20px;

    box-shadow:
        0 8px 32px rgba(0,0,0,.45);

}


/* ===========================
   Hero Title
=========================== */

.hero{

    font-size:46px;

    font-weight:700;

    color:white;

    text-align:center;

    margin-top:10px;

}


.gold{

    color:#FFD700;

}


/* ===========================
   Subtitle
=========================== */

.subtitle{

    text-align:center;

    font-size:18px;

    color:#cccccc;

    margin-bottom:30px;

}


/* ===========================
   Buttons
=========================== */

.stButton>button{

    width:100%;

    border-radius:14px;

    padding:14px;

    border:none;

    color:black;

    font-weight:700;

    background:linear-gradient(135deg,#FFD700,#F5C242);

    transition:.3s;

}

.stButton>button:hover{

    transform:translateY(-3px);

    box-shadow:0 0 20px rgba(255,215,0,.55);

}


/* ===========================
   Text Area
=========================== */

textarea{

    background:#111 !important;

    color:white !important;

    border-radius:12px !important;

}


/* ===========================
   Input Boxes
=========================== */

input{

    background:#111 !important;

    color:white !important;

}


/* ===========================
   Expanders
=========================== */

details{

    background:rgba(255,255,255,.05);

    border-radius:12px;

    padding:10px;

}


/* ===========================
   Success
=========================== */

.success-box{

    background:rgba(0,255,120,.10);

    border-left:5px solid #00ff88;

    padding:18px;

    border-radius:12px;

}


/* ===========================
   Error
=========================== */

.error-box{

    background:rgba(255,0,0,.10);

    border-left:5px solid red;

    padding:18px;

    border-radius:12px;

}


/* ===========================
   Metric Card
=========================== */

.metric{

    text-align:center;

    background:rgba(255,255,255,.05);

    border-radius:18px;

    padding:18px;

    border:1px solid rgba(255,215,0,.18);

}


/* ===========================
   Footer
=========================== */

.footer{

    text-align:center;

    color:#888;

    margin-top:60px;

    padding:20px;

}

</style>

""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------

def ask_question(question: str):
    """
    Call the /api/ask endpoint.
    """

    try:

        response = requests.post(

            f"{API_URL}/ask",

            json={

                "question": question

            },

            timeout=120,

        )

        return response.json()

    except Exception as e:

        return {

            "success": False,

            "answer": str(e)

        }


def check_contradiction(

    document1: str,

    document2: str,

    topic: str,

):
    """
    Call the /api/contradict endpoint.
    """

    try:

        response = requests.post(

            f"{API_URL}/contradict",

            json={

                "document_1": document1,

                "document_2": document2,

                "topic": topic,

            },

            timeout=120,

        )

        return response.json()

    except Exception as e:

        return {

            "success": False,

            "reasoning": str(e)

        }


def glass_card(title, body):
    """
    Render a reusable glass card.
    """

    st.markdown(

        f"""

<div class="glass">

<h3 style="color:#FFD700;">
{title}
</h3>

<p style="color:white;font-size:16px;">
{body}
</p>

</div>

""",

        unsafe_allow_html=True,

    )
# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding-top:10px;">

        <h1 style="color:#FFD700;">
            📚 RAG System
        </h1>

        <p style="color:white;">
            Document Intelligence Platform
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "💬 Ask Questions",
            "⚖️ Contradiction Checker",
        ],
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="glass">

        <h4 style="color:#FFD700;">
        System Information
        </h4>

        <p style="color:white;">

        ✓ FAISS Vector Search<br>

        ✓ Gemini LLM<br>

        ✓ Sentence Transformers<br>

        ✓ Cross Encoder Reranker<br>

        ✓ Automatic Citations<br>

        ✓ Confidence Scoring

        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="text-align:center;color:#999;font-size:13px;">
        Built using Streamlit + FastAPI
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------
# Hero Section
# -------------------------------------------------------

st.markdown(
    """
    <div class="hero">

    📚 Intelligent <span class="gold">Document QA</span>

    </div>

    <div class="subtitle">

    Ask questions across multiple policy documents using
    Retrieval-Augmented Generation (RAG), semantic search,
    reranking and Gemini AI.

    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------
# Ask Question Page Layout
# -------------------------------------------------------

if page == "💬 Ask Questions":

    glass_card(
        "Ask Questions",
        "Enter a question about your indexed documents. "
        "The system retrieves the most relevant chunks, reranks them, "
        "and generates an answer with citations.",
    )

    st.markdown("### 💬 Your Question")

    question = st.text_area(
        "",
        placeholder="Example: How many casual leaves do employees receive?",
        height=140,
        label_visibility="collapsed",
    )

    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:

        ask_button = st.button(
            "🚀 Ask Question",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass">

        <h3 style="color:#FFD700;">
        Features
        </h3>

        <ul style="color:white; line-height:2;">

        <li>Semantic Search using FAISS</li>

        <li>Cross-Encoder Reranking</li>

        <li>Gemini Answer Generation</li>

        <li>Automatic Source Citations</li>

        <li>Confidence Score</li>

        <li>Multilingual Support</li>

        <li>Hallucination Prevention</li>

        </ul>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    answer_container = st.container()
# -------------------------------------------------------
# Ask Question Logic
# -------------------------------------------------------

    if ask_button:

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("🔍 Searching documents and generating answer..."):

                response = ask_question(question)

            with answer_container:

                if response.get("success"):

                    st.markdown("---")

                    # -----------------------------------------
                    # Answer Card
                    # -----------------------------------------

                    st.markdown(
                        f"""
                        <div class="glass">

                        <h2 style="color:#FFD700;">
                        🤖 Answer
                        </h2>

                        <p style="
                            color:white;
                            font-size:18px;
                            line-height:1.8;
                        ">

                        {response.get("answer","")}

                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("")

                    # -----------------------------------------
                    # Metrics
                    # -----------------------------------------

                    confidence = response.get("confidence", 0)

                    retrieved = response.get("retrieved_chunks", 0)

                    review = response.get("needs_review", False)

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.markdown(
                            f"""
                            <div class="metric">

                            <h3 style="color:#FFD700;">
                            Confidence
                            </h3>

                            <h2 style="color:white;">
                            {confidence:.2f}
                            </h2>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col2:

                        st.markdown(
                            f"""
                            <div class="metric">

                            <h3 style="color:#FFD700;">
                            Retrieved Chunks
                            </h3>

                            <h2 style="color:white;">
                            {retrieved}
                            </h2>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with col3:

                        badge = "🟢 No"

                        if review:

                            badge = "🟡 Yes"

                        st.markdown(
                            f"""
                            <div class="metric">

                            <h3 style="color:#FFD700;">
                            Human Review
                            </h3>

                            <h2 style="color:white;">
                            {badge}
                            </h2>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("")

                    # -----------------------------------------
                    # Confidence Progress Bar
                    # -----------------------------------------

                    st.markdown("### 📊 Confidence")

                    st.progress(min(confidence, 1.0))

                    st.markdown("")

                    # -----------------------------------------
                    # Citations
                    # -----------------------------------------

                    st.markdown("## 📚 Source Citations")

                    citations = response.get("citations", [])

                    if len(citations) == 0:

                        st.info("No citations returned.")

                    else:

                        for i, cite in enumerate(citations, start=1):

                            with st.expander(
                                f"Citation {i} • {cite['file']} • Page {cite['page']}"
                            ):

                                st.markdown(
                                    f"""
**File**

`{cite['file']}`

**Page**

{cite['page']}

**Chunk**

`{cite['chunk']}`

**Snippet**

{cite['snippet']}
"""
                                )

                else:

                    st.markdown(
                        f"""
                        <div class="error-box">

                        <h3>
                        ❌ Error
                        </h3>

                        <p>

                        {response.get("answer","Unknown Error")}

                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
# -------------------------------------------------------
# Contradiction Checker Page
# -------------------------------------------------------

elif page == "⚖️ Contradiction Checker":

    glass_card(
        "Contradiction Checker",
        "Compare any two indexed documents on a specific topic and "
        "identify whether they contradict each other."
    )

    st.markdown("### 📄 Select Documents")

    col1, col2 = st.columns(2)

    documents = [
        "company_policy.pdf",
        "employee_handbook.pdf",
        "leave_policy.pdf",
        "remote_work.pdf",
        "security_policy.pdf",
        "code_of_conduct.pdf",
    ]

    with col1:

        document1 = st.selectbox(
            "Document 1",
            documents,
            key="doc1",
        )

    with col2:

        document2 = st.selectbox(
            "Document 2",
            documents,
            index=1,
            key="doc2",
        )

    st.markdown("")

    topic = st.text_input(
        "Topic",
        placeholder="Example: Casual Leave",
    )

    st.markdown("")

    col1, col2, col3 = st.columns([2,1,2])

    with col2:

        compare_button = st.button(
            "⚖️ Compare Documents",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    contradiction_container = st.container()

    # ---------------------------------------------------
    # Compare Logic
    # ---------------------------------------------------

    if compare_button:

        if not topic.strip():

            st.warning("Please enter a topic.")

        elif document1 == document2:

            st.warning("Please choose two different documents.")

        else:

            with st.spinner("Comparing documents..."):

                response = check_contradiction(
                    document1,
                    document2,
                    topic,
                )

            with contradiction_container:

                if response.get("success"):

                    conflict = response.get("conflict", False)

                    if conflict:

                        status_color = "#ff4d4d"
                        status_icon = "❌"
                        status_text = "Conflict Detected"

                    else:

                        status_color = "#00ff88"
                        status_icon = "✅"
                        status_text = "No Conflict"

                    st.markdown(
                        f"""
                        <div class="glass">

                        <h2 style="color:{status_color};">

                        {status_icon} {status_text}

                        </h2>

                        <hr>

                        <p style="
                            color:white;
                            font-size:18px;
                            line-height:1.8;
                        ">

                        <b>Topic:</b> {response.get("topic")}

                        <br><br>

                        <b>Reasoning:</b>

                        <br><br>

                        {response.get("reasoning")}

                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("")

                    st.markdown("## 📚 Supporting Citations")

                    citations = response.get("citations", [])

                    if len(citations) == 0:

                        st.info("No citations available.")

                    else:

                        for i, cite in enumerate(citations, start=1):

                            with st.expander(
                                f"Citation {i} • {cite['file']} • Page {cite['page']}"
                            ):

                                st.markdown(
                                    f"""
**File**

`{cite['file']}`

**Page**

{cite['page']}

**Chunk**

`{cite['chunk']}`
"""
                                )

                else:

                    st.markdown(
                        f"""
                        <div class="error-box">

                        <h3>

                        ❌ Comparison Failed

                        </h3>

                        <p>

                        {response.get("reasoning","Unknown Error")}

                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
           
# -------------------------------------------------------
# Footer
# -------------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="glass" style="text-align:center;">

        <h2 style="color:#FFD700;">⚡</h2>

        <h4 style="color:white;">
        Fast Retrieval
        </h4>

        <p style="color:#cccccc;">
        FAISS vector search retrieves relevant document chunks in milliseconds.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:

    st.markdown(
        """
        <div class="glass" style="text-align:center;">

        <h2 style="color:#FFD700;">🧠</h2>

        <h4 style="color:white;">
        AI Powered
        </h4>

        <p style="color:#cccccc;">
        Gemini generates grounded answers using retrieved context to reduce hallucinations.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:

    st.markdown(
        """
        <div class="glass" style="text-align:center;">

        <h2 style="color:#FFD700;">📚</h2>

        <h4 style="color:white;">
        Trusted Citations
        </h4>

        <p style="color:#cccccc;">
        Every answer includes document, page number and chunk references.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------
# Footer Information
# -------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
<div class="footer">

<hr style="border:1px solid rgba(255,215,0,.25);">

<h3 style="color:#FFD700;">
📚 RAG Document Question Answering System
</h3>

<p>

Powered by

<b>FAISS</b> •
<b>Sentence Transformers</b> •
<b>Cross Encoder Reranker</b> •
<b>Google Gemini</b> •
<b>FastAPI</b> •
<b>Streamlit</b>

</p>

<p style="font-size:14px;color:#888;">

Developed as a Retrieval-Augmented Generation (RAG) System
for Intelligent Document Question Answering.

</p>

<p style="font-size:13px;color:#666;">

© 2026 | Final Year Project

</p>

</div>

""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# Small Animation
# -------------------------------------------------------

st.markdown(
    """
<style>

/* Fade Animation */

@keyframes fadeIn {

0% {

opacity:0;

transform:translateY(15px);

}

100%{

opacity:1;

transform:translateY(0px);

}

}

.glass{

animation:fadeIn .6s ease;

}

/* Smooth Hover */

.glass:hover{

transform:translateY(-4px);

transition:.25s;

box-shadow:

0px 10px 35px rgba(255,215,0,.20);

}

/* Scroll Bar */

::-webkit-scrollbar{

width:8px;

}

::-webkit-scrollbar-track{

background:#111;

}

::-webkit-scrollbar-thumb{

background:#FFD700;

border-radius:20px;

}

/* Hide Streamlit Footer */

footer{

visibility:hidden;

}

#MainMenu{

visibility:hidden;

}

header{

visibility:hidden;

}

</style>
""",
    unsafe_allow_html=True,
)                    