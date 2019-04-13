set -u
CONN=$1

./setup_ec2.sh ${CONN}
ssh -i ~/.ssh/germany-keypair.pem -L 9999:localhost:9999 ubuntu@${CONN}
