set -u
PORT=$1

source activate tensorflow_p36
pip install livelossplot
jupyter lab --port ${PORT}
