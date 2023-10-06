import os
import requests

# URL template
url_template = "https://tone.lib.msu.edu/tone/{id}/PROXY_MP3/view"

# Directory to save downloaded files
download_dir = "mp3_files"
os.makedirs(download_dir, exist_ok=True)

# Range of IDs
start_id = 1000
end_id = 12000

for id in range(start_id, end_id + 1):
    # Construct the URL with the current ID
    current_url = url_template.format(id=id)
    
    # Extract filename
    response = requests.head(current_url)    
    if response.status_code == 200:
        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition:
            filename = content_disposition.split("filename=")[1].strip('"')
        else:
            filename = f"{id}.mp3"
        
        # Download file
        response = requests.get(current_url)
        with open(os.path.join(download_dir, filename), 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {id}.mp3 - Status code: {response.status_code}")

print("Downloaded all files")
