from fid_score.fid_score import FidScore

# data = original image folder 
# generated = fake image folder
scores = [FidScore([f"data/{i}",f"generated/{i}"], batch_size=64).calculate_fid_score() for i in range(5)]
print("Scores:")
print(*range(5),sep="\t")
print(*[f"{score:.1f}" for score in scores],sep="\t")
