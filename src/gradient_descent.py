from src.cost_functions import compute_cost

def compute_gradient(x, y, w, b): 
    m = x.shape[0]

    dj_dw = (x * (w * x + b - y)).sum() / m
    dj_db = (w * x + b - y).sum() / m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters): 
    """
    Performs gradient descent to fit w, b by taking num_iters gradient steps with lerning rate alpha

    Args: 
        x : Data, m examples
        y: target variable
        w_in, b_in (scalar): initial value for model parameters
        alpha (float): learning rate
        num_iters (int): number of iterations to run gradient descent

    Returns: 
        w (scalar): updated value of parameter after running gradient descent
        b (scalar): updated value of parameter after running gradient descent
        J_history (List): History of cost values
        p_history (List): History of parameters [w,b]
    """

    J_history = []
    p_history = []

    w = w_in
    b = b_in

    for i in range(num_iters): 
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db

    return w, b

    