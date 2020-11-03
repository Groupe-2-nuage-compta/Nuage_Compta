from src import download

dl_object = download.downloader("./data")

dl_object.data_download("dataset.zip", "https://stdatalake006.blob.core.windows.net/public/alphabet-dataset.zip")

dl_object.data_extract("dataset.zip")