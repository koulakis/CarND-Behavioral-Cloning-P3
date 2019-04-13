set -u
CONN=$1

ssh -i ~/.ssh/germany-keypair.pem ubuntu@${CONN}
