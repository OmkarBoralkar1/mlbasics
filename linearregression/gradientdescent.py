#Gradient Descent is an iterative optimization algorithm used to minimize the cost function and
# find the optimal parameters of a model. In the context of machine learning, it is commonly employed
# to update the weights of a model during training. The algorithm iteratively adjusts the model parameters
# in the direction opposite to the gradient of the cost function with respect to the parameters.
# This process continues until convergence is achieved, indicating that further adjustments do not
# significantly reduce the cost. Gradient Descent comes in various forms, such as Batch Gradient Descent, Stochastic Gradient Descent, and Mini-Batch Gradient Descent, each with its own advantages and use cases.

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Gradient Descent ek iterative optimization algorithm hai jo cost function ko kam karne aur model ke optimal parameters
# ko dhundhne ke liye istemal hota hai. Machine learning ke context mein, ise aksar model ke weights ko training
# ke dauran update karne ke liye istemal kiya jata hai. Is algorithm mein model ke parameters ko cost function
# ke gradient ke viparit disha mein iteratively adjust kiya jata hai. Ye process tab tak chalta hai jab tak
# convergence na ho jaye, jo indicate karta hai ki aur adjustments se cost mein significant kam nahi hota.
# Gradient Descent ke alag-alag roop hote hain, jaise ki Batch Gradient Descent, Stochastic Gradient Descent,
# aur Mini-Batch Gradient Descent, har ek apne laabh aur upayog ke kshetra ke sath.
import matplotlib.pyplot as plt
import numpy as np

def gradient_descent(x, y):
    m_curr = b_curr = 0
    n = len(x)  # Assuming both arrays have the same length
    learning_rate = 0.0001
    max_iterations = 10000
    costs = []  # To store the cost at each iteration
    m_values = []  # To store m_curr at each iteration
    b_values = []  # To store b_curr at each iteration

    for i in range(max_iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum({val**2 for val in (y - y_predicted)})
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        costs.append(cost)
        m_values.append(m_curr)
        b_values.append(b_curr)

        # Print values at each iteration
        print(f"Iteration {i+1}: m = {m_curr:.4f}, b = {b_curr:.4f}, cost = {cost:.4f}")

        # Check if the cost is below the threshold
        if cost < 1e-20:
            print(f"Stopping gradient descent as the cost is below 1e-20.")
            break

    # Plotting all the regression lines
    plt.scatter(x, y, color='red', label='Data points')
    for i in range(len(m_values)):
        plt.plot(x, m_values[i] * x + b_values[i], color='red')

    plt.title('All Regression Lines with Gradient Descent')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    # Plotting the cost over iterations
    plt.plot(range(1, len(costs)+1), costs, color='green')
    plt.title('Cost over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

# Ensure both x and y have the same length
x = np.array([92, 56, 88, 70, 80, 49, 65, 35, 66, 67])
y = np.array([98, 68, 81, 80, 83, 52, 66, 30, 68, 73])
gradient_descent(x, y)

# Calculate correlation between x and y
correlation = np.corrcoef(x, y)[0, 1]
print(f"Correlation between x and y: {correlation:.4f}")
