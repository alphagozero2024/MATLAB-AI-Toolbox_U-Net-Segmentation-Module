import math
import numpy as np
from np_mlp_autograd import Tensor, tensor, randn, MLP, cross_entropy, SGD


def make_blobs(n_samples=400, centers=2, random_state=42, std=1.0):
    rng = np.random.RandomState(random_state)
    angles = rng.rand(centers) * 2 * np.pi
    radii = rng.rand(centers) * 3 + 1.0
    center_pts = np.c_[radii * np.cos(angles), radii * np.sin(angles)]
    X = []
    y = []
    for i in range(n_samples):
        c = i % centers
        point = center_pts[c] + rng.randn(2) * std
        X.append(point)
        y.append(c)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)
    return X, y


def accuracy(logits: Tensor, y: np.ndarray) -> float:
    pred = logits.data.argmax(axis=1)
    return float((pred == y).mean())


def main():
    np.random.seed(0)
    X, y = make_blobs(n_samples=600, centers=3, std=0.5)
    X_t = tensor(X)
    y_t = tensor(y)  # targets shouldn't require grad by default

    model = MLP(in_features=2, hidden_sizes=[32, 32], out_features=3, activation='relu')
    optim = SGD(model.parameters(), lr=0.1)

    for epoch in range(201):
        logits = model(X_t)
        loss = cross_entropy(logits, y_t, reduction='mean')

        optim.zero_grad()
        loss.backward()
        optim.step()

        if epoch % 20 == 0:
            acc = accuracy(logits, y)
            print(f"epoch {epoch:3d} | loss={loss.item():.4f} | acc={acc*100:.2f}%")

    # inference on a few points
    test_pts = np.array([[0.0, 0.0], [2.0, 2.0], [-2.0, -1.0]], dtype=np.float64)
    test_logits = model(tensor(test_pts))
    preds = test_logits.data.argmax(axis=1)
    print("inference preds:", preds)


if __name__ == '__main__':
    main()
