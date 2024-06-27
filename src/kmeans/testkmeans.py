from ..dataset.dataset_pokemon import PokemonData
from model import KMeans

pokemon_dataset = PokemonData(image_size=(32, 32))

X_train = pokemon_dataset.getTrainX()
Y_train = pokemon_dataset.getTrainY()
X_test = pokemon_dataset.getTestX()
Y_test = pokemon_dataset.getTestY()

kmeans = KMeans(20, max_iter=100)

# Train the KMeans model
kmeans.lloyd(X_train)
print("hello")

kmeans.projection_2d(X_train, Y_train)