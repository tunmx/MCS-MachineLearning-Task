from sklearn.datasets import fetch_openml


cache_path = './cache'
mnist = fetch_openml('mnist_784', version=1, cache=True, data_home=cache_path)

X, y = mnist["data"], mnist["target"]