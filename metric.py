from nlp_tools import BLEU, Rouge

bleu = BLEU()
rouge = Rouge()

ref_file = "./data/yelp/test.txt"
system_file = "./checkpoints/yelp/non-pri-t/test.rec"

# for rouge score (simpified version)
rouge.print_score(ref_file, system_file)

# for complete version of rouge score
rouge.print_all(ref_file, system_file)

# for bleu score
# None, sm1~sm7 denotes smoothing function type
bleu.print_score(ref_file, system_file, "sm3")
bleu.print_score(ref_file, system_file, "sm5")