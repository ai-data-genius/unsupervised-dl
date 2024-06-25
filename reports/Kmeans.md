# 2D point cloud with 3 clusters

### Testing Kmeans on a 2D point cloud with 3 clusters


![img.png](img.png)

### Testing Kmeans on a 2D point cloud with 3 clusters

![img_1.png](img_1.png)


# Testing Kmeans on mnist digit dataset

### visulization of the mnist digit dataset using Kmeans

![img_2.png](img_2.png)
![img_3.png](img_3.png)
![img_4.png](img_4.png)

![img_5.png](img_5.png)

### compressing the mnist digit dataset using Kmeans

```python
def compress(self: "KMeans", image):
    image_flat = image.reshape(-1)
    image_flat = image_flat / 255.0

    id_cluster = np.argmin(np.linalg.norm(image_flat - self.centroids, axis=1))
    return id_cluster
```

### decompressing the mnist digit dataset using Kmeans

##### image before compression
![img_7.png](img_7.png)
##### image after decompression
![img_6.png](img_6.png)


### generating new images using Kmeans

![img_8.png](img_8.png)

##### This is done by interpolating between 2 random centroids of the clusters, in this case we can see it is the cluster of the digit 0 and 3


