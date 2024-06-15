## ======================= Main experiments =======================
for i in 1 5 20;
do
    for j in 'wiki-news-300d-1M.vec' 'glove.6B.300d.txt' 'random';
    do
        for k in 'gau' 'lap' 'trlap';
        do
            python3 train.py --dataset 'sst2' --eps $i --pretrained_vectors $j --method $k --dim_emb 300 --batch_size 2048
            python3 train.py --dataset 'imdb' --eps $i --pretrained_vectors $j --method $k --dim_emb 300 --batch_size 512
            python3 train.py --dataset 'ag_news' --eps $i --pretrained_vectors $j --method $k --dim_emb 300 --batch_size 512
        done
    done
done


