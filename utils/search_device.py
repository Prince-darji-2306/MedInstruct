import requests

def search_device(query, top_k=3):
    try:
        url = "https://prince-2025-all-models.hf.space/MedInsSch"
        headers = {"Content-Type": "application/json"}


        payload = {"query": query}

        response = requests.post(url, json=payload, headers=headers)
        print(response.raise_for_status())
        return response.json()['results']
    
    except Exception as e:
        return None