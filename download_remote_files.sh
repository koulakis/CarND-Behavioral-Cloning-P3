set -u
  
CONN=ubuntu@$1
   
cp_remote_file () {
    FROM=$1
	FILE=$2
	echo $FROM
    scp -i ~/.ssh/germany-keypair.pem ${CONN}:${FROM} ${FILE}
}

echo "\nCopying files..."
cp_remote_file "/home/ubuntu/cloning/*.ipynb" "./"
cp_remote_file "/home/ubuntu/cloning/*.py" "./"
