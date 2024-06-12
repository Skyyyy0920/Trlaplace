## ======================= Main experiments =======================
for i in 1 5 20;
do
    for j in 'wiki-news-300d-1M.vec' 'glove.6B.300d.txt' 'random';
    do
        for k in 'gau' 'lap' 'trlap';
        do
            for l in 'imdb' 'sst2' 'ag_news';
            do
                python3 train.py --dataset $l --eps $i --pretrained_vectors $j --method $k --dim_emb 300
            done
        done
    done
done


