import os
import requests

def make_data_from_gutenberg(start, stop):
    """creates a directory consisting of txt files, for instance: make_data_from_gutenberg(1, 15)"""
    if not os.listdir("data"):
        for book_id in range(start, stop+1):
            url = f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'

            try:
                response = requests.get(url)
                book_text = response.text
                file_path = f"data/id_{book_id}.txt"

                with open(file_path, 'w') as file:
                    file.write(book_text)
            except:
                pass
