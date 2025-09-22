def compute_cost(x, y, w, b): 
    # number of training examples
    m = x.shape[0] 
    
    f_wb = w * x + b
    cost = (f_wb - y) ** 2

    return (1 / (2 * m)) * cost.sum()