cd data

echo "Download wrime dataset"
curl -O https://raw.githubusercontent.com/ids-cv/wrime/master/wrime.tsv
# Fix column name
if sed --version > /dev/null 2>&1; then
  # Linux (GNU)
  sed -i -e "1s/Saddness/Sadness/g" wrime.tsv
else
  # Mac (BSD)
  sed -i "" -e "1s/Saddness/Sadness/g" wrime.tsv
fi
echo "Done\n"

echo "Make labeled data"
python make_data.py
echo "Done\n"

echo "Download word2vec"
curl -O http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/data/20170201.tar.bz2
tar -xvf 20170201.tar.bz2
python load_word2vec.py
echo "Done"
