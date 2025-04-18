
# Make the data directory if it doesn't exist
mkdir -p ./data/
# Download the ribosome data
wget https://drive.switch.ch/index.php/s/fWr8ZcFQiiU0OJv/download -O ribo_data.zip
unzip ribo_data.zip -d ./data/

