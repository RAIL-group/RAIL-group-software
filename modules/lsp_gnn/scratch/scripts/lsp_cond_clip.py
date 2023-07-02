from urllib import request


if __name__ == "__main__":
    # Define the remote file to retrieve
    print("Downloading CLIP encoder...")
    remote_url = "https://openaipublic.azureedge.net/clip/models/40d36571591"\
                 "3c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/"\
                 "ViT-B-32.pt"
    # Define the local filename to save data
    local_file = '/data/lsp_conditional/logs/ViT-B-32.pt'
    # Download remote and save locally
    request.urlretrieve(remote_url, local_file)
    print("Completed Downloading CLIP encoder.")
