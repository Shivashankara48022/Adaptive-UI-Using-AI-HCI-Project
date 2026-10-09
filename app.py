"""
The application develops an AI-based course recommendation system using Streamlit and Python.
Course information is processed using TF-IDF vectorization and cosine similarity techniques.
Learner preferences, including subject, difficulty level, and preferred skills, guide personalized course recommendations.
The system provides similarity scores and relevant course details through an interactive web interface.
"""


# Importing the Path utility for locating the dataset file
from pathlib import Path

# Importing pandas for loading and processing the course dataset
import pandas as pd

# Importing Streamlit for creating the interactive web application
import streamlit as st

# Importing TF-IDF for converting course text into numerical vectors
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing cosine similarity for measuring similarity between text vectors
from sklearn.metrics.pairwise import cosine_similarity


# Creating the Streamlit page configuration
st.set_page_config(
    page_title="AI Course Recommendation System",
    layout="wide"
)


# Creating the application title
st.title("AI-Based Course Recommendation System")

# Displaying the application description
st.write(
    "This intelligent system recommends educational courses "
    "based on subject interests, difficulty level, skills, "
    "and course content."
)


# Defining the required course-data columns
REQUIRED_COLUMNS = [
    "course_title",
    "subject",
    "level",
    "description",
    "skills",
    "category"
]


# Loading the course dataset
@st.cache_data
def load_data():

    # Getting the folder containing the current application
    application_folder = Path(__file__).parent

    # Creating the complete path to the course dataset
    data_path = application_folder / "courses.csv"

    # Checking whether the course dataset exists
    if not data_path.exists():
        return None, (
            f"The courses.csv file was not found.\n\n"
            f"Expected location:\n{data_path}"
        )

    try:
        # Reading the course dataset
        courses = pd.read_csv(data_path)

        # Returning the loaded dataset
        return courses, None

    except Exception as error:

        # Returning the dataset-loading error message
        return None, f"Unable to read courses.csv: {error}"


# Preparing course features for recommendation
def prepare_features(courses):

    # Creating a copy to prevent unwanted changes to the original dataset
    courses = courses.copy()

    # Converting required text columns into clean string values
    for column in REQUIRED_COLUMNS:

        # Filling missing values and converting values to strings
        courses[column] = courses[column].fillna("").astype(str)

    # Combining important course information into one text field
    courses["combined_features"] = (
        courses["course_title"] + " "
        + courses["subject"] + " "
        + courses["level"] + " "
        + courses["description"] + " "
        + courses["skills"] + " "
        + courses["category"]
    )

    # Returning the prepared dataset
    return courses


# Creating TF-IDF vectors for course features
@st.cache_data
def create_similarity_matrix(courses):

    # Creating the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    # Converting combined course information into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(
        courses["combined_features"]
    )

    # Calculating similarity between all courses
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Returning the course similarity matrix
    return similarity_matrix


# Generating course recommendations based on user preferences
def recommend_courses(
    courses,
    subject,
    level,
    skill,
    number_of_recommendations
):

    # Creating the learner preference profile
    user_profile = f"{subject} {level} {skill}"

    # Creating the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    # Combining course text and user preferences
    combined_text = courses["combined_features"].tolist()

    # Adding the learner preference profile
    combined_text.append(user_profile)

    # Creating TF-IDF representations
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    # Extracting the learner preference vector
    user_vector = tfidf_matrix[-1]

    # Extracting the course vectors
    course_vectors = tfidf_matrix[:-1]

    # Calculating similarity between learner preferences and courses
    preference_scores = cosine_similarity(
        user_vector,
        course_vectors
    ).flatten()

    # Creating a copy for recommendation processing
    results = courses.copy()

    # Adding similarity scores to the results
    results["similarity_score"] = preference_scores

    # Filtering courses by selected subject
    if subject != "All Subjects":

        # Creating subject matching conditions
        subject_matches = (
            results["subject"]
            .str.lower()
            .str.contains(
                subject.lower(),
                na=False,
                regex=False
            )
        )

        # Applying subject filtering when matches are available
        if subject_matches.any():
            results = results[subject_matches]

    # Filtering courses by selected difficulty level
    if level != "All Levels":

        # Creating level matching conditions
        level_matches = (
            results["level"]
            .str.lower()
            .str.contains(
                level.lower(),
                na=False,
                regex=False
            )
        )

        # Applying level filtering when matches are available
        if level_matches.any():
            results = results[level_matches]

    # Filtering courses by preferred skill
    if skill.strip():

        # Creating skill matching conditions
        skill_matches = (
            results["skills"]
            .str.lower()
            .str.contains(
                skill.lower(),
                na=False,
                regex=False
            )
        )

        # Applying skill filtering when matching courses exist
        if skill_matches.any():
            results = results[skill_matches]

    # Sorting courses according to similarity scores
    results = results.sort_values(
        by="similarity_score",
        ascending=False
    )

    # Selecting the requested number of recommendations
    results = results.head(number_of_recommendations)

    # Returning the recommended courses
    return results


# Loading the course dataset
courses, loading_error = load_data()


# Displaying the dataset-loading error when necessary
if loading_error:

    # Displaying the error message
    st.error(loading_error)

    # Displaying the required folder structure
    st.info(
        "Place courses.csv in the same folder as app.py."
    )

    # Stopping the application
    st.stop()


# Checking whether all required columns are available
missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in courses.columns
]


# Displaying an error when required columns are missing
if missing_columns:

    # Displaying the missing column names
    st.error(
        "The following required columns are missing from courses.csv:"
    )

    # Displaying the missing columns
    st.write(missing_columns)

    # Displaying the required column structure
    st.info(
        "Required columns: "
        + ", ".join(REQUIRED_COLUMNS)
    )

    # Stopping the application
    st.stop()


# Checking whether the dataset contains any rows
if courses.empty:

    # Displaying an empty-dataset warning
    st.warning(
        "The courses.csv file does not contain any course records."
    )

    # Stopping the application
    st.stop()


# Preparing the course dataset
courses = prepare_features(courses)


# Creating the course similarity matrix
similarity_matrix = create_similarity_matrix(courses)


# Creating the learning-preference section
st.subheader("Enter Your Learning Preferences")


# Creating two columns for subject and level selection
col1, col2 = st.columns(2)


# Creating the subject selection control
with col1:

    # Getting unique subjects from the dataset
    subjects = sorted(
        courses["subject"]
        .dropna()
        .unique()
        .tolist()
    )

    # Adding the option to include all subjects
    subject_options = ["All Subjects"] + subjects

    # Creating the subject selection box
    selected_subject = st.selectbox(
        "Select Subject",
        subject_options
    )


# Creating the difficulty-level selection control
with col2:

    # Getting unique difficulty levels from the dataset
    levels = sorted(
        courses["level"]
        .dropna()
        .unique()
        .tolist()
    )

    # Adding the option to include all levels
    level_options = ["All Levels"] + levels

    # Creating the level selection box
    selected_level = st.selectbox(
        "Select Difficulty Level",
        level_options
    )


# Creating the preferred-skill input field
selected_skill = st.text_input(
    "Enter a Preferred Skill",
    placeholder="Example: Python, SQL, Machine Learning"
)


# Creating the recommendation-count selector
number_of_recommendations = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)


# Creating the recommendation button
if st.button(
    "🔍 Recommend Courses",
    use_container_width=True
):

    # Checking whether the learner provided at least one preference
    if (
        selected_subject == "All Subjects"
        and selected_level == "All Levels"
        and not selected_skill.strip()
    ):

        # Displaying a warning for missing preferences
        st.warning(
            "Please select a subject, choose a difficulty level, "
            "or enter a preferred skill."
        )

    else:

        # Generating course recommendations
        recommendations = recommend_courses(
            courses,
            selected_subject,
            selected_level,
            selected_skill,
            number_of_recommendations
        )

        # Checking whether any recommendations were generated
        if recommendations.empty:

            # Displaying a message when no courses match
            st.warning(
                "No courses closely match the selected preferences. "
                "Try different preferences."
            )

        else:

            # Creating the recommendation-results heading
            st.subheader("Recommended Courses")

            # Displaying the number of recommended courses
            st.success(
                f"{len(recommendations)} course(s) recommended "
                "based on your preferences."
            )

            # Displaying each recommended course
            for index, (_, course) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                # Converting similarity into a percentage
                similarity_percentage = (
                    course["similarity_score"] * 100
                )

                # Creating a container for each recommendation
                with st.container():

                    # Displaying the course title
                    st.markdown(
                        f"### {index}. {course['course_title']}"
                    )

                    # Displaying the course subject
                    st.write(
                        f"**Subject:** {course['subject']}"
                    )

                    # Displaying the course difficulty level
                    st.write(
                        f"**Level:** {course['level']}"
                    )

                    # Displaying the course skills
                    st.write(
                        f"**Skills:** {course['skills']}"
                    )

                    # Displaying the similarity percentage
                    st.write(
                        f"**Similarity:** "
                        f"{similarity_percentage:.2f}%"
                    )

                    # Displaying the course description
                    st.write(
                        course["description"]
                    )

                    # Separating individual recommendations
                    st.divider()


# Creating the methodology information section
with st.expander("About the Recommendation Method"):

    # Explaining the content-based filtering approach
    st.write(
        "The system uses content-based filtering to recommend "
        "courses with characteristics similar to the learner's "
        "preferences."
    )

    # Explaining the TF-IDF technique
    st.write(
        "TF-IDF (Term Frequency-Inverse Document Frequency) "
        "converts course-related text into numerical vectors."
    )

    # Explaining cosine similarity
    st.write(
        "Cosine similarity measures the similarity between the "
        "learner preference profile and course representations."
    )

    # Explaining the course information used by the system
    st.write(
        "Course title, subject, level, description, skills, "
        "and category information are combined for text-based "
        "recommendation."
    )