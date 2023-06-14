"""
This module contains the code for scraping research papers from arXiv.org using the arXiv API.

@Date: June 6th, 2023
@Author: Yassine RODANI
"""

import arxiv
import pandas as pd

from tqdm import tqdm

from constants import PATH_DATA_BASE


query_keywords = [
    "\"image segmentation\"",
    "\"self-supervised learning\"",
    "\"representation learning\"",
    "\"image generation\"",
    "\"object detection\"",
    "\"transfer learning\"",
    "\"transformers\"",
    "\"adversarial training",
    "\"generative adversarial networks\"",
    "\"model compressions\"",
    "\"image segmentation\"",
    "\"few-shot learning\"",
    "\"natural language\"",
    "\"graph\"",
    "\"colorization\"",
    "\"depth estimation\"",
    "\"point cloud\"",
    "\"structured data\"",
    "\"optical flow\"",
    "\"reinforcement learning\"",
    "\"super resolution\"",
    "\"attention\"",
    "\"tabular\"",
    "\"unsupervised learning\"",
    "\"semi-supervised learning\"",
    "\"explainable\"",
    "\"radiance field\"",
    "\"decision tree\"",
    "\"time series\"",
    "\"molecule\"",
    "\"large language models\"",
    "\"llms\"",
    "\"language models\"",
    "\"image classification\"",
    "\"document image classification\"",
    "\"encoder\"",
    "\"decoder\"",
    "\"multimodal\"",
    "\"multimodal deep learning\"",
]


def query_with_keywords(query, client) -> tuple:
    """
    Query the arXiv API for research papers based on a specific query and filter results by selected categories.
    
    Args:
        query (str): The search query to be used for fetching research papers from arXiv.
    
    Returns:
        tuple: A tuple containing three lists - terms, titles, and abstracts of the filtered research papers.
        
            terms (list): A list of lists, where each inner list contains the categories associated with a research paper.
            titles (list): A list of titles of the research papers.
            abstracts (list): A list of abstracts (summaries) of the research papers.
            urls (list): A list of URLs for the papers' detail page on the arXiv website.
    """

    # Create a search object with the query and sorting parameters.
    search = arxiv.Search(
        query=query,
        max_results=6000,
        sort_by=arxiv.SortCriterion.LastUpdatedDate
    )

    # Initialize empty lists for terms, titles, abstracts, and urls.
    terms = []
    titles = []
    abstracts = []
    urls = []

    # For each result in the search...
    for res in tqdm(client.results(search), desc=query):
        # Check if the primary category of the result is in the specified list.
        if res.primary_category in ["cs.CV", "stat.ML", "cs.LG", "cs.AI"]:
            # If it is, append the result's categories, title, summary, and url to their respective lists.
            terms.append(res.categories)
            titles.append(res.title)
            abstracts.append(res.summary)
            urls.append(res.entry_id)

    # Return the four lists.
    return terms, titles, abstracts, urls


def main() -> None:

    client = arxiv.Client(num_retries=20, page_size=500)

    all_titles = []
    all_abstracts = []
    all_terms = []
    all_urls = []

    print("\n→ Querying arXiv API for research papers...\n")

    for query in query_keywords:
        terms, titles, abstracts, urls = query_with_keywords(query, client)
        all_titles.extend(titles)
        all_abstracts.extend(abstracts)
        all_terms.extend(terms)
        all_urls.extend(urls)

    print("\n→ Writing results to CSV file...\n")

    arxiv_data = pd.DataFrame({
        'titles': all_titles,
        'abstracts': all_abstracts,
        'terms': all_terms,
        'urls': all_urls
    })

    arxiv_data.to_csv(PATH_DATA_BASE / "data.csv", index=False)

    print("\n→ Scraping completed!\n ")


if __name__ == '__main__':
    main()
