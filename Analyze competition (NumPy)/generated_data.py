import numpy as np
# shape (20, 10, 4): 20 - days, 10 - sportsman, 4 - indicators (spint, pushups, jumps, pulse)
def training_data(seed = 42):
    np.random.seed(42)
    sprint = np.random.uniform(8.5, 15.0, (20, 10)).round(2) #running (seconds)
    pushups = np.random.randint(20, 60, (20, 10)) #pushups per minute
    jumps = np.random.uniform(5, 8, (20, 10)).round(2) #jumps (metres)
    pulse = np.random.randint(40, 120, (20, 10)) #hear rate
    data_overall = np.stack((sprint, pushups, jumps, pulse), axis = 2)
    return data_overall

