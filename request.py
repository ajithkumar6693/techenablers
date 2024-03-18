
import requests

url = "https://hackathonteam57.search.windows.net/indexes/azureblob-index/docs"
params = {
    "api-version": "2023-10-01-Preview",
    "search": "Tell me about wells fargo",
    "select": "*",
    "$top": 5,
    "queryLanguage": "en-us",
    "queryType": "semantic",
    "semanticConfiguration": "hackathonteam57",
    "$count": "true",
    "speller": "lexicon",
    "answers": "extractive|count-3",
    "captions": "extractive|highlight-false"
}
headers = {
    "Api-Key": "ZNMud1mRRzsN09BvOXx6A3JzpPg3xWClpvMcdV9SQ7AzSeDRmNXp"
}

try:
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    # Process response here
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
