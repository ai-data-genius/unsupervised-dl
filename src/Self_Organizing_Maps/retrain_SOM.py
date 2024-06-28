from src.dataset.dataset_pokemon import PokemonData
from src.Self_Organizing_Maps.model_PM import SOM
from math import sqrt

pokemon_dataset = PokemonData(image_size=(32, 32))

X_train = pokemon_dataset.getTrainX()
Y_train = pokemon_dataset.getTrainY()
X_test = pokemon_dataset.getTestX()
Y_test = pokemon_dataset.getTestY()


map_size = 5 * sqrt(X_train.shape[0])
print(f"Map size: {map_size}")
# Find the multiplication of map size, for example if map size is 64, the x and y will be 8*8
x = 20
y = 20
input_size = 1024
som = SOM(X_train,x, y, input_size, num_epochs=100, learning_rate=0.02, NW=2, colors=True)

som.load_weights('models/model_20_20_1_200_0.02.npy')

X_train = X_train.reshape(X_train.shape[0], -1) / 255
som.train(X_train)

som.save_weights()

som.save_weights()