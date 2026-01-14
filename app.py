

import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

@st.cache_data
def compute_similarity(movies):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    return cosine_similarity(vectors)

similarity = compute_similarity(movies)

#st.title('🎬 Movie Recommendation System')
st.markdown(
    """
    <h1 style='text-align: center;'>🎬 Movie Recommendation System</h1>
    <p style='text-align: center; color: gray;'>
    Content-based recommender using cosine similarity
    </p>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("🎬 About")
st.sidebar.info(
    """
    **Movie Recommendation System**

    - Content-based filtering
    - Cosine similarity
    - Streamlit app

    Built by **Arbaaz**
    """
)

col1, col2 = st.columns([1, 2])

with col1:
    selected_movie_name = st.selectbox(
        "🎥 Select a movie",
        movies['title'].values
    )
    #recommend_btn = st.button("✨ Recommend")

with col2:
    st.subheader("🍿 Recommended Movies")

#selected_movie_name = st.selectbox(
 #   'Which movie do you want?',
  #  movies['title'].values
#)

#if st.button('Recommend'):
 #   recommendations = recommend(selected_movie_name)
    #for i in recommendations:
     #   st.write("👉", i)
if st.button("Recommend"):
    recommendations = recommend(selected_movie_name)

    for movie in recommendations:
        st.markdown(
            """
            <style>
            .movie-card {
                background: linear-gradient(135deg, #111827, #1f2937);
                color: #f9fafb;
                padding: 14px 18px;
                border-radius: 12px;
                margin-bottom: 12px;
                font-size: 18px;
                font-weight: 500;
                box-shadow: 0 4px 10px rgba(0,0,0,0.25);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

if recommend_btn:
    with st.spinner("🔍 Finding best matches..."):
        recommendations = recommend(selected_movie_name)

