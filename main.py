import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
from utils.augment_image import download_and_augment_images

# Base URL for scraping
base_url = "https://www.adl.org/resources/hate-symbols/search"
training_output_dir = "./training_data/images"
image_counter = 0
# Headers to mimic a browser visit
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}


def scrape_page(url, output_dir):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")

        # Find all div elements with class 'global-search__row'
        search_rows = soup.find_all("div", class_="global-search__row")

        for row in search_rows:
            link = row.find("a", class_="search-result__link")
            if link:
                href = link.get("href")
                full_url = urljoin(base_url, href)

                # Fetch the linked page
                get_images = requests.get(full_url, headers=headers)
                if get_images.status_code == 200:
                    image_soup = BeautifulSoup(get_images.content, "html.parser")
                    search_images = image_soup.find_all(
                        "div", class_="hate-symbol__media--additional-images"
                    )

                    for image in search_images:
                        img_tags = image.find_all("img")
                        for img_tag in img_tags:
                            if img_tag:
                                # print(img_tag.get("data-src"))

                                img_src = img_tag.get("data-src")
                                img_full_url = urljoin("https://www.adl.org", img_src)
                                print(f"Image URL: {img_full_url}")
                                global image_counter

                                download_and_augment_images(
                                    img_full_url, output_dir, image_counter
                                )
                                image_counter += 1

                else:
                    print(f"Failed to fetch linked page: {full_url}")

                text = link.text.strip()
                print(f"Text: {text}, Link: {full_url}")

        # Find the next page link
        next_page = soup.find("li", class_="pager__item pager__item--next")
        if next_page:
            next_link = next_page.find("a")
            if next_link:
                next_url = urljoin(base_url, next_link.get("href"))
                print(f"Moving to next page: {next_url}")
                scrape_page(next_url, output_dir)
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")


# Start scraping from the first page
scrape_page(base_url, training_output_dir)
