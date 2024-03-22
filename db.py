import pickle

with open('data/faces_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)