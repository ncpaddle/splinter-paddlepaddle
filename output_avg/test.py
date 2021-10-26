import pickle

a = pickle.load(open('examples16_average.json', 'rb'))
b = pickle.load(open('examples128_average.json', 'rb'))
c = pickle.load(open('examples1024_average.json', 'rb'))
#


print(a)
print(b)
print(c)