ls my-videos | xargs -I {} mkdir -p my-videos-center/{}
ls my-videos | xargs -I {} mkdir -p my-videos-center/{}/IMG

ls my-videos | xargs -I{} sh -c "cp my-videos/{}/IMG/center*.jpg my-videos-center/{}/IMG/"
ls my-videos | xargs -I{} sh -c "cp my-videos/{}/driving_log.csv my-videos-center/{}/"
