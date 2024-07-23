import os
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.imsdb.com"

def get_movie_links():
    response = requests.get(f"https://www.imsdb.com/all-scripts.html")
    soup = BeautifulSoup(response.text, 'html.parser')
    links = [a['href'] for a in soup.select('p a[href^="/Movie Scripts/"]')]
    return links

def get_script_details(link):
    try:
        response = requests.get(f"https://www.imsdb.com{link}")
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.find('title').text.split(' Script at IMSDb.')[0]
        script_link = soup.find('a', text=f'Read "{title}" Script')['href']

        script_response = requests.get(f"https://www.imsdb.com{script_link}")
        script_soup = BeautifulSoup(script_response.text, 'html.parser')
        script_text = script_soup.pre.text if script_soup.pre else ""

        return {
            'title': title,
            'script': script_text
        }
    except:
        return None

def save_script(title, script):
    # Create a directory to store scripts
    os.makedirs('scripts', exist_ok=True)
    # File path with safe title
    file_path = os.path.join('scripts', f"{title.replace(' ', '_').replace('/', '_')}.txt")
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(script)
    print(f"Saved script: {title}\n")

def main():
    movie_links = get_movie_links()
    for link in movie_links:
        print(f"scraping https://www.imsdb.com{link}")
        script_details = get_script_details(link)
        if script_details:
            save_script(script_details['title'], script_details['script'])
        else:
            print("Script not found\n")

if __name__ == "__main__":
    main()
