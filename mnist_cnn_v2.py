import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# -----------------------------
# Load MNIST (modern, reliable)
# -----------------------------
(images, labels), (test_images, test_labels) = mnist.load_data()

# Flatten labels to avoid sklearn warnings
labels = labels.ravel()
test_labels = test_labels.ravel()

# Flatten images for KNN
images_flat = images.reshape(len(images), -1)
test_images_flat = test_images.reshape(len(test_images), -1)

# -----------------------------
# Condensed Nearest Neighbor (CNN) Prototype Selection
# -----------------------------
#def prototype(images_flat, labels, M):
 #  Classic CNN prototype selection.
  #  Adds a point to the prototype set if the current prototype set misclassifies it.
   # """
    #all_indices = list(range(len(images_flat)))
    #random.shuffle(all_indices)

    # Start with one random prototype
    #proto_indices = [all_indices[0]]

    #for idx in all_indices[1:]:
     #   if len(proto_indices) >= M:
      #      break

        # Build temporary KNN on current prototypes
       # proto_imgs = images_flat[proto_indices]
        #proto_labs = labels[proto_indices]

        #knn = KNeighborsClassifier(n_neighbors=1)
        #knn.fit(proto_imgs, proto_labs)

        # Predict current sample
        #pred = knn.predict(images_flat[idx].reshape(1, -1))[0]

        # If misclassified → add to prototype set
       # if pred != labels[idx]:
         #   proto_indices.append(idx)

    #return proto_indices[:M]

def prototype_fast(images_flat, labels, M):
    """
    Fast Condensed Nearest Neighbor (CNN) prototype selection.
    Uses vectorized distance computations instead of retraining KNN repeatedly.
    Produces identical results to the classic CNN algorithm.
    """

    n = len(images_flat)
    all_indices = np.arange(n)
    np.random.shuffle(all_indices)

    # Start with one prototype
    proto_indices = [all_indices[0]]

    # Precompute squared norms for fast distance calculation
    img_norms = np.sum(images_flat**2, axis=1)

    for idx in all_indices[1:]:
        if len(proto_indices) >= M:
            break

        # Compute distances from current image to all prototypes
        P = np.array(proto_indices)
        dists = (
            img_norms[P]
            + img_norms[idx]
            - 2 * np.dot(images_flat[P], images_flat[idx])
        )

        nearest_proto = P[np.argmin(dists)]
        predicted_label = labels[nearest_proto]

        # If misclassified → add to prototype set
        if predicted_label != labels[idx]:
            proto_indices.append(idx)

    return proto_indices[:M]


# -----------------------------
# Evaluate prototype sets
# -----------------------------
def evaluate_prototype_set(proto_indices):
    proto_imgs = images_flat[proto_indices]
    proto_labs = labels[proto_indices]

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(proto_imgs, proto_labs)

    return knn.score(test_images_flat, test_labels)

# -----------------------------
# Run prototype selection
# -----------------------------
print("Building prototype set of size 1000...")
proto_1000 = prototype_fast(images_flat, labels, 1000)
acc_1000 = evaluate_prototype_set(proto_1000)
print("Accuracy (1000):", acc_1000)

print("Building prototype set of size 5000...")
proto_5000 = prototype_fast(images_flat, labels, 5000)
acc_5000 = evaluate_prototype_set(proto_5000)
print("Accuracy (5000):", acc_5000)

print("Building prototype set of size 10000...")
proto_10000 = prototype_fast(images_flat, labels, 10000)
acc_10000 = evaluate_prototype_set(proto_10000)
print("Accuracy (10000):", acc_10000)

# -----------------------------
# Plot your error bars
# -----------------------------
protomean = (0.1088, 0.0595, 0.0417)
randommean = (0.1149, 0.0655, 0.0564)
protostd = (0.001757, 0.002042, 0.002145)
randomstd = (0.00494, 0.002432, 0.00275)

N = len(randommean)
ind = np.arange(N)
width = 0.4

fig, ax = plt.subplots()
rects1 = ax.bar(ind, protomean, width, color='MediumSlateBlue',
                yerr=protostd, error_kw={'ecolor':'Tomato','linewidth':2})
rects2 = ax.bar(ind+width, randommean, width, color='Tomato',
                yerr=randomstd, error_kw={'ecolor':'MediumSlateBlue','linewidth':2})

ax.set_ylim([0, 0.15])
ax.set_ylabel('Error Rate')
ax.set_title('Error Bar')
ax.set_xticks(ind + width)
ax.set_xticklabels(('M = 1000', 'M = 5000', 'M = 10000'))
ax.legend([rects1, rects2], ['Prototype', 'Random'])

plt.show()
