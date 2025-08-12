import numpy as np
from generated_data import training_data

def the_best_athlete(data):
    mean_result_for_all_days = np.mean(data, axis = 0).round(2)
    the_best_in_sprint = np.argmin(mean_result_for_all_days[:, 0])
    the_best_in_pushups = np.argmax(mean_result_for_all_days[:, 1])
    the_best_in_jumps = np.argmax(mean_result_for_all_days[:, 2])
    return the_best_in_sprint, the_best_in_pushups, the_best_in_jumps


def record_of_day(data):
    spint_pushups_jumps_record = np.max(data, axis = 0)
    return spint_pushups_jumps_record


def stable_result_sprint(data):
    sprint_data = data[:, :, 0]
    var_per_athlete = np.var(sprint_data, axis = 0)
    the_most_stable_ind = np.argmin(var_per_athlete)
    the_most_stable = var_per_athlete[the_most_stable_ind]
    return the_most_stable_ind, the_most_stable


def quickest_athlete(data):
    sprint_data = data[:, :, 0]
    mask = np.all(sprint_data < 14, axis = 1)
    return np.where(mask)[0]


def longest_jump(data):
    jumps_data = data[:, :, 2]
    mask = np.any(jumps_data > 7.5, axis = 1)
    return np.where(mask)[0]


def labels_sprint(data):
    sprint_data = data[:, :, 0]
    conditions = np.full(sprint_data.shape, 'Average', dtype = object)
    slow = sprint_data > 13.5
    fast = sprint_data <= 11
    conditions[slow] = 'Slow'
    conditions[fast] = 'Fast'
    return conditions

def labels_heart_rate(data):
    pulse_data = data[:, :, 3]
    conditions = np.full(pulse_data.shape, 'Normal', dtype = object)
    low = pulse_data < 60
    high = pulse_data > 100
    conditions[low] = 'Low'
    conditions[high] = 'High'
    return conditions

