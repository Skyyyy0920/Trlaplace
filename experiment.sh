## ======================= Main experiments =======================
for i in 1 5 20;
do
    for j in 'wiki-news-300d-1M.vec' 'glove.6B.300d.txt' 'random';
    do
        for k in 'imdb' 'ag_news' 'sst2';
        do
            for l in 'gau' 'lap' 'trlap';
            do
                python3 train.py --dataset $k --eps $i --pretrained_vectors $j --method $l --dim_emb 300
            done
        done
    done
done


