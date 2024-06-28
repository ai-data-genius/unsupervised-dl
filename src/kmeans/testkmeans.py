from src.kmeans.model import KMeans
from src.dataset.dataset_pokemon import PokemonData
import matplotlib.pyplot as plt
import numpy as np

kmeans = KMeans(n_clusters=100, max_iter=100)
kmeans.load_weights("models/model_400_100.npy")

pokemon_dataset = PokemonData(image_size=(32, 32))

X_train = pokemon_dataset.getTrainX()
Y_train = pokemon_dataset.getTrainY()
X_test = pokemon_dataset.getTestX()
Y_test = pokemon_dataset.getTestY()

fig, axes = plt.subplots(30, 2, figsize=(10, 50))  # Create a 10x2 grid of subplots


for i in range(30):
    axes[i, 0].imshow(X_test[i].reshape(32, 32, 3))
    axes[i, 0].axis('off')  # Hide axes
    axes[i, 0].set_title(f"Original {i+1}")

    compressed = kmeans.compress(X_test[i])
    decompressed = kmeans.decompress(compressed)

    # Ensure decompressed is a NumPy array
    decompressed = np.array(decompressed, dtype=np.float32)

    axes[i, 1].imshow(decompressed)
    axes[i, 1].axis('off')  # Hide axes
    axes[i, 1].set_title(f"Decompressed {i+1}")

plt.tight_layout()
# Save the figure
plt.savefig('comparison_plot.png', bbox_inches='tight')

# Display the plot
plt.show()