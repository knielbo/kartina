# KARTINA - Image Search for Culture Analytics #



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

For running in virtual environment (recommended) and assuming python3.6+ is installed.

```
sudo pip3 install virtualenv
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate
```

### Installing

Clone repository and install requirements

```
git clone https://github.com/knielbo/kartina.git
pip install -r requirements.txt
```

To run train model and generate graph

```
./main.sh
```

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

```
./test.sh
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With


## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## Versioning


## Authors
Kristoffer L. Nielbo  
Ross D. Kristensen-McLachlan

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Adrian Rosebrock, pyimagesearch
