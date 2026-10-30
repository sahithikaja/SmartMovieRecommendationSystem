import streamlit as st
import pandas as pd
import pickle
import requests
import time

from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(
    page_title="Smart Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Smart Movie Recommendation System")
st.write(
    "Get movie recommendations based on content, mood and personal preferences."
)

movies = pickle.load(open("movies.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
vectors = pickle.load(open("vectors.pkl", "rb"))
TMDB_ACCESS_TOKEN = st.secrets["TMDB_ACCESS_TOKEN"]

session = requests.Session()

# Helps with the network problem we encountered earlier
session.trust_env = False

headers = {
    "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}",
    "accept": "application/json"
}

def tmdb_request(url, params=None, retries=3):

    for attempt in range(retries):

        try:

            response = session.get(
                url,
                headers=headers,
                params=params,
                timeout=20
            )

            if response.status_code == 200:
                return response.json()

            elif response.status_code == 429:
                time.sleep(2)

            else:
                return None

        except requests.exceptions.RequestException:

            if attempt < retries - 1:
                time.sleep(2)

    return None
def search_tmdb(movie_name):

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "query": movie_name,
        "include_adult": False
    }

    data = tmdb_request(url, params)

    if data and data.get("results"):
        return data["results"][0]

    return None
def get_movie_details(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    params = {
        "append_to_response": "credits,keywords"
    }

    return tmdb_request(url, params)
def get_tmdb_similar(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}/similar"

    params = {
        "page": 1
    }

    data = tmdb_request(url, params)

    if data:
        return data.get("results", [])

    return []

mood_data = {

    "😊 Happy": {
        "genres": [35, 12, 16, 10751],
        "keywords": [
            "fun",
            "funny",
            "happy",
            "comedy",
            "adventure",
            "family"
        ]
    },

    "😢 Emotional": {
        "genres": [18, 10749],
        "keywords": [
            "emotional",
            "love",
            "family",
            "relationship",
            "life"
        ]
    },

    "😨 Scared": {
        "genres": [27, 53, 9648],
        "keywords": [
            "horror",
            "fear",
            "mystery",
            "dark",
            "survival",
            "thriller"
        ]
    },

    "❤️ Romantic": {
        "genres": [10749, 18],
        "keywords": [
            "love",
            "romance",
            "relationship",
            "couple"
        ]
    },

    "🤣 Want to Laugh": {
        "genres": [35],
        "keywords": [
            "comedy",
            "funny",
            "humor",
            "laugh"
        ]
    },

    "🧠 Thought-Provoking": {
        "genres": [878, 9648, 18, 53],
        "keywords": [
            "science",
            "space",
            "future",
            "mystery",
            "psychological",
            "mind"
        ]
    },

    "😌 Relaxed": {
        "genres": [16, 35, 10751, 10402],
        "keywords": [
            "family",
            "animation",
            "music",
            "fun",
            "peaceful"
        ]
    },

    "🔥 Excited": {
        "genres": [28, 12, 53],
        "keywords": [
            "action",
            "adventure",
            "fight",
            "thriller",
            "chase"
        ]
    }
}

def calculate_mood_score(movie, selected_mood):

    if selected_mood == "No Preference":
        return 0

    mood_info = mood_data[selected_mood]

    score = 0

    # Check genres
    movie_genres = movie.get("genre_ids", [])

    for genre in movie_genres:

        if genre in mood_info["genres"]:
            score += 1

    # Check title and overview
    text = (
        str(movie.get("title", ""))
        + " "
        + str(movie.get("overview", ""))
    ).lower()

    for word in mood_info["keywords"]:

        if word in text:
            score += 0.5

    return score

def calculate_local_mood_score(movie, selected_mood):

    if selected_mood == "No Preference":
        return 0

    mood_info = mood_data[selected_mood]

    text = str(movie.get("tags", "")).lower()

    score = 0

    for word in mood_info["keywords"]:

        if word in text:
            score += 1

    return score
def runtime_match(runtime, runtime_preference):

    # If runtime is missing
    if pd.isna(runtime) or runtime is None:

        return 0

    try:
        runtime = float(runtime)

    except:

        return 0

    if runtime_preference == "Any":
        return 0

    if runtime_preference == "Under 90 minutes":

        return 1 if runtime < 90 else 0

    if runtime_preference == "90–120 minutes":

        return 1 if 90 <= runtime <= 120 else 0

    if runtime_preference == "120–150 minutes":

        return 1 if 120 < runtime <= 150 else 0

    if runtime_preference == "Over 150 minutes":

        return 1 if runtime > 150 else 0

    return 0

def normalize_scores(scores):

    if not scores:
        return []

    maximum = max(scores)
    minimum = min(scores)

    if maximum == minimum:

        return [1.0 for _ in scores]

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


def get_local_recommendations(
    movie_title,
    selected_mood,
    runtime_preference
):

    movie_index = movies[
        movies["title"].str.lower()
        == movie_title.lower()
    ].index

    if len(movie_index) == 0:
        return []

    index = movie_index[0]

    query_vector = vectors[index]

    scores = cosine_similarity(
        query_vector,
        vectors
    ).flatten()

    similar_indices = scores.argsort()[::-1]

    recommendations = []

    for i in similar_indices:

        # Don't recommend selected movie
        if i == index:
            continue

        movie = movies.iloc[i]

        runtime = movie.get(
            "runtime",
            0
        )

        recommendations.append({

            "title": movie["title"],

            "id": str(movie["id"]),

            "overview": str(
                movie.get("overview", "")
            ),

            "language": str(
                movie.get(
                    "original_language",
                    ""
                )
            ),

            "rating": float(
                movie.get(
                    "vote_average",
                    0
                )
            ),

            "runtime": runtime,

            "similarity": float(
                scores[i]
            ),

            "mood_score":
                calculate_local_mood_score(
                    movie,
                    selected_mood
                ),

            "runtime_match":
                runtime_match(
                    runtime,
                    runtime_preference
                ),

            "source": "Your Dataset"

        })

        if len(recommendations) >= 10:
            break

    return recommendations

def get_external_recommendations(
    tmdb_movie_id,
    selected_mood
):

    similar_movies = get_tmdb_similar(
        tmdb_movie_id
    )

    recommendations = []

    for movie in similar_movies[:10]:

        recommendations.append({

            "title": movie.get(
                "title",
                "Unknown"
            ),

            "id": str(
                movie.get("id", "")
            ),

            "overview": movie.get(
                "overview",
                ""
            ),

            "language": movie.get(
                "original_language",
                ""
            ),

            "rating": float(
                movie.get(
                    "vote_average",
                    0
                )
            ),

            # TMDB /similar does not reliably
            # provide runtime
            "runtime": None,

            "similarity": 0,

            "mood_score":
                calculate_mood_score(
                    movie,
                    selected_mood
                ),

            "runtime_match": 0,

            "source": "TMDB"

        })

    return recommendations

def remove_duplicates(
    recommendations,
    selected_movie_id
):

    seen_ids = set()
    seen_titles = set()

    final = []

    for movie in recommendations:

        movie_id = str(
            movie.get("id", "")
        )

        title = str(
            movie.get("title", "")
        ).lower().strip()

        # Don't recommend selected movie
        if movie_id == str(selected_movie_id):
            continue

        # Duplicate ID
        if movie_id in seen_ids:
            continue

        # Duplicate title
        if title in seen_titles:
            continue

        seen_ids.add(movie_id)
        seen_titles.add(title)

        final.append(movie)

    return final
def rank_recommendations(
    recommendations,
    preferred_language,
    minimum_rating,
    runtime_preference
):

    if not recommendations:
        return []

    count = len(recommendations)

    for index, movie in enumerate(
        recommendations
    ):

        movie["content_score"] = (
            count - index
        ) / count


    mood_scores = [
        movie["mood_score"]
        for movie in recommendations
    ]

    normalized_mood = normalize_scores(
        mood_scores
    )

    for movie, score in zip(
        recommendations,
        normalized_mood
    ):

        movie["normalized_mood"] = score

    for movie in recommendations:

        content_score = movie[
            "content_score"
        ]

        mood_score = movie[
            "normalized_mood"
        ]

        # Language
        if preferred_language == "Any":

            language_score = 0

        elif (
            movie["language"]
            == preferred_language
        ):

            language_score = 1

        else:

            language_score = 0

        # Runtime
        runtime_score = movie[
            "runtime_match"
        ]

        # Rating
        rating_score = (
            movie["rating"] / 10
        )

        movie["final_score"] = (

            0.55 * content_score
            + 0.20 * mood_score
            + 0.10 * language_score
            + 0.10 * runtime_score
            + 0.05 * rating_score

        )

    preferred = []
    others = []

    for movie in recommendations:

        language_match = (
            preferred_language == "Any"
            or movie["language"]
            == preferred_language
        )

        rating_match = (
            movie["rating"]
            >= minimum_rating
        )

        runtime_match_value = (
            runtime_preference == "Any"
            or movie["runtime_match"] == 1
        )

        if (
            language_match
            and rating_match
            and runtime_match_value
        ):

            preferred.append(movie)

        else:

            others.append(movie)

    # Sort
    preferred.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    others.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    final = preferred.copy()

    for movie in others:

        if len(final) >= 5:
            break

        final.append(movie)

    return final[:5]

def generate_reason(
    movie,
    selected_movie,
    selected_mood,
    preferred_language,
    minimum_rating,
    runtime_preference
):

    reasons = []

    reasons.append(
        f"it is similar to your selected movie "
        f"'{selected_movie}'"
    )

    if selected_mood != "No Preference":

        if movie["mood_score"] > 0:

            mood_name = selected_mood.split(
                " ",
                1
            )[-1]

            reasons.append(
                f"it matches your "
                f"{mood_name.lower()} mood"
            )


    if (
        runtime_preference != "Any"
        and movie["runtime"] is not None
    ):

        if movie["runtime_match"] == 1:

            reasons.append(
                f"its runtime of "
                f"{int(movie['runtime'])} minutes "
                f"matches your preferred duration"
            )

    if preferred_language != "Any":

        if movie["language"] == preferred_language:

            reasons.append(
                f"it matches your preferred "
                f"{preferred_language} language"
            )
    if movie["rating"] >= minimum_rating:

        reasons.append(
            f"it has a rating of "
            f"{movie['rating']:.1f}/10"
        )


    if len(reasons) == 1:

        return (
            "Recommended because "
            + reasons[0]
            + "."
        )

    elif len(reasons) == 2:

        return (
            "Recommended because "
            + reasons[0]
            + " and "
            + reasons[1]
            + "."
        )

    else:

        return (
            "Recommended because "
            + ", ".join(reasons[:-1])
            + ", and "
            + reasons[-1]
            + "."
        )

st.subheader(
    "🎯 Tell us what you want to watch"
)


movie_name = st.text_input(
    "🎬 Enter any movie name",
    placeholder="Example: Interstellar"
)


col1, col2, col3 = st.columns(3)

with col1:

    selected_mood = st.selectbox(
        "😊 What is your mood?",
        [
            "No Preference",
            "😊 Happy",
            "😢 Emotional",
            "😨 Scared",
            "❤️ Romantic",
            "🤣 Want to Laugh",
            "🧠 Thought-Provoking",
            "😌 Relaxed",
            "🔥 Excited"
        ]
    )

with col2:

    preferred_language = st.selectbox(
        "🌐 Preferred Language",
        [
            "Any",
            "en",
            "hi",
            "te",
            "ta",
            "ml",
            "kn"
        ]
    )


with col3:

    minimum_rating = st.slider(
        "⭐ Minimum Rating",
        1.0,
        10.0,
        7.0,
        0.5
    )

runtime_preference = st.selectbox(
    "⏱️ Preferred Movie Duration",
    [
        "Any",
        "Under 90 minutes",
        "90–120 minutes",
        "120–150 minutes",
        "Over 150 minutes"
    ]
)


if st.button(
    "🎬 Get Recommendations",
    type="primary"
):

    if not movie_name.strip():

        st.warning(
            "Please enter a movie name."
        )

    else:

        with st.spinner(
            "Finding the best movies for you..."
        ):


            tmdb_movie = search_tmdb(
                movie_name
            )

            if tmdb_movie is None:

                st.error(
                    "Movie not found on TMDB. "
                    "Please check the movie name."
                )

                st.stop()

            selected_title = tmdb_movie.get(
                "title",
                movie_name
            )

            selected_id = tmdb_movie.get(
                "id"
            )

            local_recommendations = (
                get_local_recommendations(
                    selected_title,
                    selected_mood,
                    runtime_preference
                )
            )

            external_recommendations = (
                get_external_recommendations(
                    selected_id,
                    selected_mood
                )
            )

            all_recommendations = (
                local_recommendations
                + external_recommendations
            )

            all_recommendations = (
                remove_duplicates(
                    all_recommendations,
                    selected_id
                )
            )
            final_recommendations = (
                rank_recommendations(
                    all_recommendations,
                    preferred_language,
                    minimum_rating,
                    runtime_preference
                )
            )
        st.subheader(
            "🍿 Your Personalized Recommendations"
        )

        if not final_recommendations:

            st.warning(
                "No recommendations found."
            )

        else:

            for number, movie in enumerate(
                final_recommendations,
                start=1
            ):

                st.markdown(
                    f"### {number}. {movie['title']}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.write(
                        f"⭐ Rating: "
                        f"{movie['rating']:.1f}/10"
                    )

                with col2:

                    st.write(
                        f"🌐 Language: "
                        f"{movie['language']}"
                    )

                with col3:

                    if movie["runtime"] is not None:

                        st.write(
                            f"⏱️ Runtime: "
                            f"{int(movie['runtime'])} min"
                        )

                    else:

                        st.write(
                            "⏱️ Runtime: Not available"
                        )

                st.write(
                    f"📚 Source: "
                    f"{movie['source']}"
                )

                reason = generate_reason(
                    movie,
                    selected_title,
                    selected_mood,
                    preferred_language,
                    minimum_rating,
                    runtime_preference
                )

                st.info(
                    f"💡 **Why this movie?** "
                    f"{reason}"
                )

                if movie["overview"]:

                    st.write(
                        movie["overview"]
                    )

                st.divider()