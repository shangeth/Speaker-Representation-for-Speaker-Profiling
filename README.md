# Learning Speaker Representation with Semi-supervised Learning approach for Speaker Profiling

![](assets/framework.PNG)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

## Download the Dataset

```
# LibriSpeech Train-Clean-360(Unsupervised Learning Dataset)
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
tar -xvzf train-clean-360.tar.gz -C 'path to libri data folder'
```
```
# TIMIT(Supervised Learning Dataset)
wget wget https://data.deepai.org/timit.zip
unzip timit.zip -d 'path to timit data folder'
```

## Prepare Dataset for Training

```
# TIMIT Dataset
python TIMIT/prepare_data.py --path='path to timit data folder'
```
```
# LibriSpeech Dataset
python LibriSpeech/prepare_data.py --path='path to libri data folder'
```


## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)