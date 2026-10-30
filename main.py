import kagglehub

# Download latest version
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000",output_dir="data/raw/",force_download=True)

print("Path to dataset files:", path)