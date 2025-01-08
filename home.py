import streamlit as st

st.title("Suggestions")
st.header("This is a Suggestion Platform which suggests Movies, TV series and Anime based on your preference.")


st.sidebar.page_link("home.py", label="Home")
st.sidebar.page_link("pages/movies.py", label="Movies")
st.sidebar.page_link("pages/anime.py", label="Anime")

st.divider()  # 👈 Draws a horizontal rule

# Layout for the movie and anime images
col1, col2 = st.columns(2)

with col1:
    st.image("Images/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg", width=158)
    st.page_link("pages/movies.py", label="Movies", icon="🎥")

with col2:
    st.image("Images/73245.jpg", width=150)
    st.page_link("pages/anime.py", label="Anime", icon="🌇")