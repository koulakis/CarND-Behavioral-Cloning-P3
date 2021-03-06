set -u

CONN=ubuntu@$1

exec_remote () {
	COMMAND=$1
	ssh -i ~/.ssh/germany-keypair.pem ${CONN} bash -c "'${COMMAND}'"
}

mk_remote_dir () {
	DIR_PATH=$1
	exec_remote "mkdir ${DIR_PATH}"
}

cp_file_remote () {
	FILE=$1
	TO=$2
	scp -i ~/.ssh/germany-keypair.pem ${FILE} ${CONN}:${TO}
}

echo "\nCreating directories..."
mk_remote_dir "cloning"
mk_remote_dir "cloning/my-videos-center"
mk_remote_dir "cloning/models"
mk_remote_dir "cloning/cloning"

echo "\nCopying files..."
cp_file_remote "*.ipynb" "/home/ubuntu/cloning/"
cp_file_remote "*.py" "/home/ubuntu/cloning/"
cp_file_remote "cloning/*" "/home/ubuntu/cloning/cloning/"
cp_file_remote "setup_notebook.sh" "/home/ubuntu/" 
cp_file_remote "*.sh" "/home/ubuntu/cloning/"
exec_remote "chmod +x cloning/*.sh"
exec_remote "chmod +x *.sh"

echo "\nSyncing data from S3..."
exec_remote "aws s3 sync s3://behavioral-cloning/ cloning/my-videos-center/"
exec_remote "aws s3 sync s3://behavioral-cloning-models/ cloning/models/"
