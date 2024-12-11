import gdown

# URL of the model on Google Drive
url = 'https://drive.google.com/uc?export=download&id=1uvAp_I30bpXBt3nHQfQiow7MoGZXsJYx'
output = 'plant_disease_model.h5'

# Download model with SSL verification disabled
gdown.download(url, output, quiet=False, verify=False)
