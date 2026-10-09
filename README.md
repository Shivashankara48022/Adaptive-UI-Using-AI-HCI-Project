# AI-Based Adaptive Course Recommendation System

## Project Overview

The AI-Based Adaptive Course Recommendation System is a web application developed using Python and Streamlit to help users discover courses that match their learning interests, preferred subjects, and skill development goals. The system uses artificial intelligence techniques, including Term Frequency-Inverse Document Frequency (TF-IDF) and cosine similarity, to analyze course descriptions and identify relevant learning opportunities. The application provides an interactive interface that allows users to select preferences, explore recommended courses, express interest in particular courses, and reset their selections when needed.

## Project Objectives

The main objective of the project is to improve the online learning experience by providing personalized course recommendations. Instead of requiring users to search through an entire course catalog, the application ranks courses according to their selected preferences and the similarity between course information and user interests. The project also demonstrates how adaptive user interface principles can support user-centered design, improve information discovery, and make educational platforms easier to navigate.

## Technologies Used

The application is developed using Python as the primary programming language. Streamlit is used to create the interactive web interface, while pandas is used to load, organize, and process course information from a CSV file. Scikit-learn provides the TF-IDF vectorization and cosine similarity functionality used to compare course descriptions and calculate relevance scores. These technologies work together to create a simple and accessible course recommendation application.

## Project Structure

The project contains four main files. The app.py file manages the Streamlit interface, preference selection, recommendation display, and user interactions. The recommender.py file contains the recommendation logic used to process course information and rank relevant courses. The courses.csv file stores the sample course dataset, including course titles, subjects, difficulty levels, descriptions, and associated skills. The requirements.txt file lists the Python packages required to install and run the application.

## How the Application Works

The application begins by loading course information from the CSV dataset. Users can select their preferred subjects, difficulty levels, or learning interests through the interface. The recommendation component processes the available course descriptions using TF-IDF, which converts text into numerical representations. Cosine similarity is then used to compare the textual representations and estimate how closely courses match the selected learning interests. The system ranks the courses based on relevance and presents the recommendations through the Streamlit interface. Users can explore the results, indicate interest in a course, and reset their preferences when necessary.

## Installation and Execution

To run the application locally, first install Python and ensure that Python is available from the command line. Download or clone the project repository and open a terminal in the project directory. Install the required packages by executing python -m pip install -r requirements.txt. After the installation finishes, start the application by running python -m streamlit run app.py. Streamlit will display a local URL, usually http://localhost:8501, which can be opened in a web browser to access the course recommendation interface.

## Using the Application

After opening the application, users can review the available course preferences and select options that reflect their learning goals. The system uses the selected information to rank relevant courses from the sample dataset. Users can review course titles, subjects, difficulty levels, descriptions, and related skills to decide which courses best meet their needs. The interface also provides course-interest interactions and a reset option so users can change their selections and explore different recommendations.

## Adaptive User Interface Features

The application demonstrates adaptive interface concepts by using user preferences to influence the displayed course recommendations. The recommendation list changes according to the selected learning interests, helping users focus on relevant educational content. The interface organizes course information in a readable format and provides interactive controls for exploring options. Preference and interaction information may be maintained during the current application session, depending on the implementation in app.py. These features illustrate how personalization can support a more relevant and user-centered learning experience.

## Dataset Description

The courses.csv file contains sample educational course information. The course_title column identifies each course, while subject represents its academic area. The difficulty column indicates the expected learning level, and the description column provides information about the course content. The skills column lists skills associated with each course. These fields supply the textual and categorical information used to organize courses and generate recommendations. Because the project uses a sample dataset, the quality and variety of recommendations depend on the information available in the file.

## Testing and Limitations

The application can be tested by selecting different subjects, changing difficulty preferences, exploring recommended courses, interacting with course-interest controls, and resetting the selections. Testing should confirm that the dataset loads correctly, the interface displays the available controls, and recommendations are generated without errors. The current implementation is a demonstration rather than a fully validated commercial recommendation system. Its results depend on the sample dataset and the text-similarity method, and it does not establish recommendation accuracy through a formal evaluation. User preferences may also be limited to the active session if persistent storage has not been implemented.

## Future Enhancements

Future development could include expanding the course dataset, adding more detailed learner profiles, and collecting explicit feedback about recommendation quality. Persistent storage could be introduced to maintain user preferences between sessions. Additional recommendation methods, such as collaborative filtering or hybrid recommendation models, could also be evaluated. Accessibility improvements, responsive interface design, and systematic usability testing would help make the application more useful to a wider range of learners.

## AI Tools Disclosure

AI assistance may be used during development to support brainstorming, explain Python concepts, troubleshoot programming errors, and improve documentation. The project implementation and report should be reviewed to ensure that the descriptions accurately reflect the submitted code. Any AI tools used should be disclosed according to the course instructor's requirements and the university's academic integrity guidelines.

