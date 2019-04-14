set -u
CONN=$1
PORT=$2

./setup_ec2.sh ${CONN}
ssh -i ~/.ssh/germany-keypair.pem -L ${PORT}:localhost:${PORT} ubuntu@${CONN}
