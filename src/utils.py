from torchvision import transforms
import cv2
import numpy as np 
import torch


index_to_labels = [ 
        "Speed Limit 5 km/h", 
        "Speed Limit 15 km/h", 
        "Speed Limit 30 km/h", 
        "Speed Limit 40 km/h", 
        "Speed Limit 50 km/h", 
        "Speed Limit 60 km/h", 
        "Speed Limit 70 km/h", 
        "Speed Limit 80 km/h", 
        "No car allowed"
    ]

def pytorch_switch(tensor_image):
    return tensor_image.permute(1, 2, 0)


def to_pytorch(tensor_image):
    return torch.from_numpy(tensor_image).permute(2, 0, 1)


transform = transforms.Compose([
  transforms.Resize((128,128)), 
  transforms.ToTensor(),
])


def l2(adv_patch, orig_patch):
    assert adv_patch.shape == orig_patch.shape
    return np.sum((adv_patch - orig_patch) ** 2)


def sh_selection(n_queries, iter):
    t = max((float(n_queries - iter) / n_queries), 0) * .75
    return t


def update_location(loc_new, h_i, h, s):
    loc_new += np.random.randint(low=-h_i, high=h_i + 1, size=(2,))
    loc_new = np.clip(loc_new, 0, h - s)
    return loc_new


def render(x, w):
    phenotype = np.ones((w, w, 3))
    radius_avg = (phenotype.shape[0] + phenotype.shape[1]) / 2 / 6
    for row in x:
        overlay = phenotype.copy()
        cv2.circle(
            overlay,
            center=(int(row[1] * w), int(row[0] * w)),
            radius=int(row[2] * radius_avg),
            color=(int(row[3] * 255), int(row[4] * 255), int(row[5] * 255)),
            thickness=-1,
        )
        alpha = row[6]
        phenotype = cv2.addWeighted(overlay, alpha, phenotype, 1 - alpha, 0)

    return phenotype/255.


def mutate(soln, mut):
    """Mutates specie for evolution.

    Args:
        specie (species.Specie): Specie to mutate.

    Returns:
        New Specie class, that has been mutated.
        :param soln:
    """
    new_specie = soln.copy()

    # Randomization for Evolution
    genes = soln.shape[0]
    length = soln.shape[1]
    y = np.random.randint(0, genes)
    change = np.random.randint(1, length + 1)

    selection = np.random.choice(length, size=change, replace=False)

    if np.random.rand() < mut:
        new_specie[y, selection] = np.random.rand(len(selection))
    else:
        new_specie[y, selection] += (np.random.rand(len(selection)) - 0.5) / 6
        new_specie[y, selection] = np.clip(new_specie[y, selection], 0, 1)

    return new_specie


