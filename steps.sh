pip install kaggle
pip install pandas

mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download chrisfilo/urbansound8k
unzip urbansound8k.zip -d urbansound8k
rm urbansound8k.zip 

