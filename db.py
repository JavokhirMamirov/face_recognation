import pickle

with open('data/faces.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)