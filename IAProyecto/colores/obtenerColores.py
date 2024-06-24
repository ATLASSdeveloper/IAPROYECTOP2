import requests
from bs4 import BeautifulSoup
import os
import time

def get_google_images(query, color_folder, num_images):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"}
    img_count = 0
    img_index = 0

    if not os.path.exists(color_folder):
        os.makedirs(color_folder)

    while img_count < num_images:
        url = f"https://www.google.com/search?hl=en&tbm=isch&q={query}&start={img_index}"
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error fetching images for query '{query}': {response.status_code}")
            break
        
        soup = BeautifulSoup(response.text, 'html.parser')

        image_tags = soup.find_all("img")

        if not image_tags:
            print(f"No more images found for query '{query}'.")
            break

        for image_tag in image_tags:
            img_url = image_tag.get("src")
            if img_url and img_url.startswith('http'):
                img_data = requests.get(img_url).content
                img_file_path = os.path.join(color_folder, f"{query.replace(' ', '_')}_{img_count+1}.jpg")
                with open(img_file_path, "wb") as img_file:
                    img_file.write(img_data)
                img_count += 1
                if img_count >= num_images:
                    break

        img_index += 20  # Move to the next page
        time.sleep(2)  # Add a delay to avoid being blocked

    print(f"{img_count} images downloaded for query '{query}' in folder '{color_folder}'")

# Define the colors and their variants
color_variants = {
    "blue": ["blue", "dark blue", "light blue", "navy", "sky blue"],
    "yellow": ["yellow", "dark yellow", "light yellow", "gold", "lemon"],
    "green": ["green", "dark green", "light green", "olive", "lime"],
    "orange": ["orange", "dark orange", "light orange", "amber", "peach"],
    "purple": ["purple", "dark purple", "light purple", "violet", "lavender"],
    "black": ["black", "dark black", "light black", "charcoal", "jet black"],
    "white": ["white", "dark white", "light white", "ivory", "snow"],
    "gray": ["gray", "dark gray", "light gray", "slate", "ash"],
    "brown": ["brown", "dark brown", "light brown", "chocolate", "tan"]
}

# Total number of images per color
total_images_per_color = 2000

# Download images for each color and its variants
for color, variants in color_variants.items():
    num_variants = len(variants)
    images_per_variant = total_images_per_color // num_variants

    for variant in variants:
        print(f"Downloading images for '{variant}'...")
        get_google_images(f"{variant} objects", color, images_per_variant)
