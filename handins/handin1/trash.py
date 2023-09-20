""" rows, cols = X.shape
        
        #Sum used for the loss-function
        sum = 0
        
        #Sum used for the gradient
        sum2 = 0
        for i in range(rows):
            sum += np.log(1 + np.exp(-y[i] * w.T @ X[i]))

            j = 0
            for j in range(cols):
                sum2 += (1 / (1 + np.exp(-y[i] * w.T @ X[i]))) * np.exp(-y[i] * w.T @ X[i]) * (-y[i] * X[i, j])
                j = j
            grad[j] = sum2 / rows

        #Compute the cost of th loss function
        cost = sum / rows """