# Contributing

If you notice an error, please don't hesitate to open an issue.

## Development setup

```sh
# update
sudo apt update
# install Python for ensuring that tests can be run
sudo apt install python3-pip \
  python3.7 python3.7-dev python3.7-distutils python3.7-venv \
  python3.8 python3.8-dev python3.8-distutils python3.8-venv \
  python3.9 python3.9-dev python3.9-distutils python3.9-venv \
  python3.10 python3.10-dev python3.10-distutils python3.10-venv \
  python3.11 python3.11-dev python3.11-distutils python3.11-venv
  python3.12 python3.12-dev python3.12-distutils python3.12-venv
  python3.13 python3.13-dev python3.13-venv
# install pipenv for creation of virtual environments
python3.8 -m pip install pipenv --user

# check out repo
git clone https://github.com/jasminsternkopf/mel_cepstral_distance.git
cd mel_cepstral_distance
# create virtual environment
python3.8 -m pipenv install --dev
```

## Running the tests

```sh
# first, install the tool (see "Development setup")
# then, navigate into the directory of the repo
cd mel_cepstral_distance
# activate environment
python3.8 -m pipenv shell
# run tests
tox
```
