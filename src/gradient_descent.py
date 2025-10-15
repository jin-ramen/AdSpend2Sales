from src.cost_functions import compute_cost
import copy

def compute_gradient(x, y, w, b): 
    m = x.shape[0]

    f_wb = w*x + b
    dj_dw = (x * (f_wb - y)).sum() / m
    dj_db = (f_wb - y).sum() / m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters): 
    m = len(x)

    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters): 
        dj_dw, dj_db = compute_gradient(x, y, w_in, b_in)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000: 
            cost = compute_cost(x, y, w, b)
            J_history.append(cost)

        if i % math.ceil(num_iters/10) == 0: 
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history
        
    