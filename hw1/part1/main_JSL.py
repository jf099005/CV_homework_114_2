dogs = []
for i in range(4):
    dog = cv2.subtract(gaussian[i+1], gaussian[i])
    dogs.append(dog)

# ---- Normalize and save ----
for i, dog in enumerate(dogs):
    # Normalize to [0,255] for visualization
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    dog_norm = dog_norm.astype(np.uint8)

    filename = f"DoG1-{i+1}.png"
    cv2.imwrite(filename, dog_norm)
    print("DoG raw min/max:", dog.min(), dog.max())
    print("DoG normalized min/max:", dog_norm.min(), dog_norm.max())
dogs = []
for i in range(4):
    dog = cv2.subtract(gaussian[i+1], gaussian[i])
    dogs.append(dog)

# ---- Normalize and save ----
for i, dog in enumerate(dogs):
    # Normalize to [0,255] for visualization
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    dog_norm = dog_norm.astype(np.uint8)

    filename = f"DoG1-{i+1}.png"
    cv2.imwrite(filename, dog_norm)
    print("DoG raw min/max:", dog.min(), dog.max())
    print("DoG normalized min/max:", dog_norm.min(), dog_norm.max())
