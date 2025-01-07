import os
import requests
from typing import List
from dotenv import load_dotenv

def configure():
    load_dotenv()

class imgDown:
    def __init__(self, key: str):
        self.key = key
        self.base_url = "https://pixabay.com/api/"

    def grab_img(self, search: str, imageType: str, minWidth: int, minHeight: int, perPage: int, limit: int):
        total_imgs = []
        page = 1
        search = search.replace(' ', '+')

        while len(total_imgs) < limit:
            url = f"{self.base_url}?key={self.key}&q={search}&image_type={imageType}&min_width={minWidth}&min_height={minHeight}&page={page}&per_page={perPage}"

            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error fetching data: {response.status_code}")

            data = response.json()
            if 'hits' not in data or not data['hits']:
                print("No imgs found")
                break

            total_imgs.extend(hit['webformatURL'] for hit in data['hits'])

            page += 1

        return total_imgs[:limit]


    def download_img(self, img_urls: List[str], save_dir: str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i, url in enumerate(img_urls):
            try:
                ext = os.path.splitext(url.split("/")[-1])[1]
                if ext.lower() not in [".png", ".jpeg", ".jpg"]:
                    ext = ".jpg"

                file_name = f"image_{i + 1}{ext}"
                file_path = os.path.join(save_dir, file_name)

                response = requests.get(url)
                response.raise_for_status()

                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded: {file_name}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {url: {e}}")


def main():
    configure()
    style1Search = "cyberpunk"
    style2Search = "cosmic"
    contentSearch = "japanese art"
    imageType = "illustration"
    minWidth = 200
    minHeight = 200
    perPage = 5
    limit = 30

    style1Down = imgDown(os.getenv('API_KEY'))
    style2Down = imgDown(os.getenv('API_KEY'))
    contentDown = imgDown(os.getenv('API_KEY'))

    style1Urls = style1Down.grab_img(style1Search, imageType, minWidth, minHeight, perPage, limit)
    if style1Urls:
        print("Style1 Image URLs: ")
        for url in style1Urls:
            print(url)
        style1Down.download_img(style1Urls, "style1_images")

    style2Urls = style2Down.grab_img(style2Search, imageType, minWidth, minHeight, perPage, limit)
    if style2Urls:
        print("Style2 Image URLs: ")
        for url in style2Urls:
            print(url)
        style1Down.download_img(style2Urls, "style2_images")

    contentUrls = contentDown.grab_img(contentSearch, imageType, minWidth, minHeight, perPage, limit)
    if contentUrls:
        print("Content Image URLs: ")
        for url in contentUrls:
            print(url)
        contentDown.download_img(contentUrls, "content_images")

if __name__ == "__main__":
    main()