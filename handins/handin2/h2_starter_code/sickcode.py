def sickcode():
# Forward pass
        u1 = X @ W1
        z1 = u1 + b1
        a1 = relu(z1)

        u2 = a1 @ W2
        z2 = u2 + b2
        a2 = softmax(z2)

        # Cost
        # cost = -np.sum(labels * np.log(a2)) + c * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        a2_correct = np.choose(y, a2.T)
        a2_correct = np.clip(a2_correct, 1e-15, 1 - 1e-15)
        cost = np.mean(-np.log(a2_correct)) + c * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        
        # Backwards pass
        d_z2 = a2 - labels
        d_b2 = np.mean(d_z2, axis=0, keepdims=True)
        d_w2 = (a1.T @ d_z2) / batch_size + c * 2 * W2

        assert d_z2.shape == z2.shape
        assert d_b2.shape == b2.shape
        assert d_w2.shape == W2.shape

        d_a1 = d_z2 @ W2.T
        d_z1 = derive_relu(d_a1, z1)
        d_b1 = np.mean(d_z1, axis=0, keepdims=True)
        d_w1 = (X.T @ d_z1) / batch_size + c * 2 * W1

        assert d_z1.shape == z1.shape
        assert d_a1.shape == a1.shape
        assert d_b1.shape == b1.shape
        assert d_w1.shape == W1.shape

        return cost, {'d_w1': d_w1, 'd_w2': d_w2, 'd_b1': d_b1, 'd_b2': d_b2}