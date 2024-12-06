import pickle


# open
with open('work_dirs/vis/tr3d/1/tr3d03_result.pkl', 'rb') as f :
    data = pickle.load(f)
print(data)