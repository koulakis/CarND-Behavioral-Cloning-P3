ls my-videos | xargs -I {} mkdir -p my-videos-small/{}
ls my-videos | xargs -I {} mkdir -p my-videos-small/{}/IMG

ls my-videos | xargs -I{} sh -c "cp my-videos/{}/IMG/center*.jpg my-videos-small/{}/IMG/"
ls my-videos | xargs -I{} sh -c "cp my-videos/{}/driving_log.csv my-videos-small/{}/"
