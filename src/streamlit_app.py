"""
This script contains the main function to run the PaperWhiz Streamlit app.

@Date: June 6th, 2023
@Author: Yassine RODANI
"""

from pathlib import Path
import pandas as pd
import numpy as np

import streamlit as st
import pickle
from dotenv import load_dotenv

import torch
from sentence_transformers import SentenceTransformer, util

from trubrics.integrations.streamlit import FeedbackCollector
from trubrics.cli.main import init
import hopsworks

from constants import PATH_EMBEDDINGS, PATH_SENTENCES, PATH_DATA_BASE

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load hopsworks API key and Trubrics credentials from .env file

load_dotenv()

try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('Set environment variable HOPSWORKS_API_KEY')



######################## Helper functions ########################



@st.cache_resource
def init_trubrics() -> FeedbackCollector:
    """
    Initialize an environment and authenticate with the Trubrics platform account.
    
    This function connects to the Trubrics platform using the project name stored in the `secrets.toml` file as an environment variable.
    The Trubrics account authentication is assumed to be set with 'single_user' mode.
    
    Environment Variables:
        TRUBRICS_CONFIG_EMAIL: Email for Trubrics account, stored in the `secrets.toml` file.
        TRUBRICS_CONFIG_PASSWORD: Password for Trubrics account, stored in the `secrets.toml` file.
        TRUBRICS_PROJECT_NAME: Project name to initiate on the Trubrics platform, also stored in the `secrets.toml` file in a [general] section.
        
    Returns:
        FeedbackCollector: A FeedbackCollector object authenticated with Trubrics platform.

    """

    with st.spinner("Connecting to the Trubrics platform..."):
        init(project_name=os.environ['TRUBRICS_PROJECT_NAME'])
    return FeedbackCollector(trubrics_platform_auth="single_user")


def get_paper_abstracts() -> dict:
    """
    Fetch paper abstracts from a CSV file.

    Returns:
        dict: A dictionary mapping paper titles to their abstracts.
    """

    df = pd.read_csv(PATH_DATA_BASE / "filtered_data.csv")
    paper_abstracts = dict(zip(df['titles'], df['abstracts']))

    return paper_abstracts


def get_paper_urls() -> dict:
    """
    Fetch paper URLs from a CSV file.

    Returns:
        dict: A dictionary mapping paper titles to their URLs.
    """

    df = pd.read_csv(PATH_DATA_BASE / "filtered_data.csv")
    paper_urls = dict(zip(df['titles'], df['urls']))

    return paper_urls


def get_sentences_data(path: str) -> list:
    """
    Load sentence data from a pickle file.
    
    Args:
        path (str): The location of the pickle file to read from.

    Returns:
        list: A list of sentences loaded from the pickle file.
    """

    with open(path, 'rb') as f:
        sentences = pickle.load(f)

    return sentences


def get_embeddings_data(path: str) -> np.array:
    """
    Load embedding data from a pickle file.
    
    Args:
        path (str): The location of the pickle file to read from.

    Returns:
        numpy.array: An array of embeddings loaded from the pickle file.
    """

    with open(path, 'rb') as f:
        embeddings = pickle.load(f)

    return embeddings


@st.cache_resource
def connect_to_hopsworks() -> tuple :
    """
    Connects to Hopsworks feature store using a provided API key.

    Returns:
        tuple: A tuple containing the Hopsworks project and feature store objects.
    """
    with st.spinner("Connecting to Hopsworks Feature Store..."):
        try:
            project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
            fs = project.get_feature_store()
            
            print("Connected to the Hopsworks Feature Store")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    return project, fs


@st.cache_resource
def get_data_from_hopsworks(project, feature_store) -> pd.DataFrame:
    """
    Fetches data from the 'papers_info' feature group.

    Returns:
        pd.DataFrame: A DataFrame containing the features from the 'papers_info'
                      feature group. If the connection or data fetch fails, it 
                      will return an empty DataFrame.
    """
    
    try:
        feature_store = project.get_feature_store()
        print("Connected to the Hopsworks Feature Store")
    except Exception as e:
        print(f"An error occurred: {e}")

    with st.spinner("Fetching data from Hopsworks Feature Store..."):
        feature_group = feature_store.get_feature_group("papers_info", version=1)
    # Pull the feature group as a Pandas DataFrame
    df = feature_group.read()
    return df


def get_model() -> SentenceTransformer:
    """
    This function creates a SentenceTransformer model using the 'all-MiniLM-L6-v2' configuration and returns it.

    Returns:
        SentenceTransformer: A SentenceTransformer model with 'all-MiniLM-L6-v2' configuration.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model



############################### Main function ##############################



def main() -> None:
    html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """

    st.set_page_config(page_title="PaperWhiz",
                       page_icon="📜", layout="centered")
    st.title("PaperWhiz: A simple app to help you find relevant ML papers to read")

    # Create a feedback collector
    collector = init_trubrics()

    # Connect to Hopsworks Feature Store
    project, fs = connect_to_hopsworks()

    with st.sidebar:
        st.markdown("""
        # About 
        Locating relevant scientific papers to read can be a daunting task, especially
        when delving into a new field. 
        
        
        To address this, I have developed a NLP-based recommender system for machine learning papers 
        This system is not only tailored to aid researchers and data scientists, but also eager students, 
        by recommending germane papers specifically aligned with their interests.
        """)
        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),
                    unsafe_allow_html=True)

        # TODO: Add 'By Abstracts' option to the sidebar
        option = st.selectbox(
            'How would you like to calculate similarity?',
            ('Using titles', 'Using abstracts'), index=0)

        if option == 'Using abstracts':
            st.info(
                "This option is coming soon. 'Using titles' has been selected by default.")
            option = 'Using titles'  # set to default or another option

        st.markdown("""
        # How does it work
        Simply enter a topic of interest, select the preferred method for calculating similarity, and the system will recommend relevant papers for you to read.
        
        For instance, if you want to read about Transformers in Computer Vision, type:

        "I'm interested in papers about Transformers in Computer Vision."
        """)
        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),
                    unsafe_allow_html=True)
        st.markdown("""
        Made by [yassine-rd](https://github.com/yassine-rd) with ❤️
        """,
                    unsafe_allow_html=True,
                    )

    # Load sentence and embedding data if the user chooses to calculate similarity by titles
    if option == 'Using titles':
        sentences = get_sentences_data(PATH_SENTENCES / "sentences.pkl")
        embeddings = get_embeddings_data(PATH_EMBEDDINGS / "embeddings.pkl")

    # TODO: Load sentence and embedding data if the user chooses to calculate similarity by abstracts

    # Load the SentenceTransformer model
    model = get_model()
    
    # Fetch data from Hopsworks Feature Store
    # data = get_data_from_hopsworks()

    # Load paper abstracts and URLs
    paper_abstracts = get_paper_abstracts()
    paper_urls = get_paper_urls()
    
    # Initializations for session state variables
    if "paper_recommendations" not in st.session_state:
        st.session_state["paper_recommendations"] = []
    if "feedback_submitted" not in st.session_state:
        st.session_state["feedback_submitted"] = False

    # Include a form to receive a topic of interest from the user
    form = st.form(key="my_form")
    topic_of_interest = form.text_input(label="Enter your topic of interest here 👇",
                                        placeholder="I'm interested in papers about Transformers in Computer Vision")
    submit = form.form_submit_button(label="Recommend papers")

    # If user submits a topic of interest
    if submit:
        # Check if input is not empty
        if topic_of_interest.strip() == "":
            st.error("Please enter a topic of interest.")
        else:
            # Calculate cosine similarity between the user's input embedding and the pre-calculated embeddings of paper titles
            cosine_scores = util.cos_sim(
                embeddings, model.encode(topic_of_interest))

            # Get the indices of the top 4 papers with highest cosine scores
            topic_of_interest = torch.topk(
                cosine_scores, dim=0, k=4, sorted=True)

            # Store the top 4 paper titles into session state to keep them persistent
            st.session_state["paper_recommendations"] = [
                sentences[i.item()] for i in topic_of_interest.indices]

            # Reset feedback submission status in session state to False when new recommendations are made
            st.session_state["feedback_submitted"] = False

    # If there are paper recommendations in session state
    if st.session_state["paper_recommendations"]:
        # Display recommended papers
        st.subheader("Here are some papers you might be interested in reading")
        for paper_title in st.session_state["paper_recommendations"]:
            with st.expander(paper_title):
                paper_abstract = paper_abstracts.get(
                    paper_title, "Abstract not found")
                st.markdown(paper_abstract)
                paper_url = paper_urls.get(paper_title, "URL not found")
                st.markdown("Read the full paper here: " + paper_url)

        # If feedback has not been submitted for current recommendations
        if not st.session_state["feedback_submitted"]:
            # Display feedback form
            st.subheader("How satisfied are you with these recommendations?")
            feedback = collector.st_feedback(
                feedback_type="faces", open_feedback_label="Optionally, provide some detail regarding the accuracy of these recommendations.")

            # If user submits feedback
            if feedback:
                # Update feedback submission status in session state to True, hiding the feedback form until new recommendations are made
                st.session_state["feedback_submitted"] = True


if __name__ == "__main__":
    main()
