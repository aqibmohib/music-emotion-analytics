import streamlit as st

def apply_theme():
    st.markdown("""
    <style>

    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Page background */
    .stApp {
        background: radial-gradient(circle at top, #0f172a, #020617);
        color: #e5e7eb;
    }

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 3rem;
    }

    /* Headings */
    h1 {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    h2 {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 2rem;
    }

    h3 {
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.9;
    }

    /* Navigation bar */
    .nav {
        display: flex;
        gap: 1rem;
        margin-bottom: 2.5rem;
    }

    .nav button {
        flex: 1;
        padding: 1.2rem;
        font-size: 1rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.03);
        color: #e5e7eb;
        transition: all 0.25s ease;
    }

    .nav button:hover {
        border-color: #38bdf8;
        box-shadow: 0 0 0 1px #38bdf8 inset, 0 0 30px rgba(56,189,248,0.15);
        transform: translateY(-2px);
    }/* Card-style navigation buttons */
.stButton > button {
    height: 110px;
    text-align: left;
    padding: 1.2rem 1.4rem;
    font-size: 1rem;
    line-height: 1.4;
}
def apply_theme():
    st.markdown("""
    <style>
    .stButton > button {
        height: 120px;
        padding: 1.4rem 1.6rem;
        text-align: left;
        font-size: 1rem;
        line-height: 1.45;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.15);
        background: linear-gradient(145deg, #0b1220, #050914);
        color: #ffffff;
        transition: all 0.25s ease-in-out;
    }

    .stButton > button:hover {
        border: 1px solid #4da6ff;
        box-shadow: 0 0 12px rgba(77,166,255,0.6);
        transform: translateY(-2px);
    }
    </style>
    
<style>
/* REMOVE underline / divider below nav buttons */
.nav-divider,
.nav-underline,
.nav-bar,
hr {
    display: none !important;
}

/* Remove :after underline effect */
button::after {
    display: none !important;
    content: none !important;
}
</style>
""", unsafe_allow_html=True)


    </style>
    """, unsafe_allow_html=True)
