import numpy as np

from generated_data import training_data
from analyze import (
    the_best_athlete,
    record_of_day,
    stable_result_sprint,
    quickest_athlete,
    longest_jump,
    labels_sprint,
    labels_heart_rate
)


data = training_data()
the_best_in_sprint, the_best_in_pushups, the_best_in_jumps = the_best_athlete(data)
records = record_of_day(data)
stable_var_ind, stable_var = stable_result_sprint(data)
fast_days = quickest_athlete(data)
long_jump_days = longest_jump(data)

tips_sprint = labels_sprint(data)
fast_count = np.sum(tips_sprint == 'Fast')
slow_count = np.sum(tips_sprint == 'Slow')
medium_count = np.sum(tips_sprint == 'Average')

tips_pulse = labels_heart_rate(data)
high_count = np.sum(tips_pulse == 'High')
low_count = np.sum(tips_pulse == 'Low')
normal_count = np.sum(tips_pulse == 'Normal')



print('The best athletes:')
print(f'Sprint: Athlete #{the_best_in_sprint + 1}')
print(f'Push-ups: Athlete #{the_best_in_pushups + 1}')
print(f'Jump: Athlete #{the_best_in_jumps + 1}')
print()

print('All-time records:')
print(f'Sprint (seconds): {records[:, 0].min()}')
print(f'Push-ups (times/min): {int(records[:, 1].max())}')
print(f'Jump (m): {records[:, 2].max()}')
print()

print('Pulse rate categories:')
print(f'{high_count} athletes had a pulse with high values (> 100).')
print(f'{normal_count} athletes had a pulse with average values (>= 60 & < 100).')
print(f'{low_count} athletes had a pulse with low values (< 60).')
print()

print('Categories by running speed:')
print(f'Fastest athletes (< 11 seconds): {fast_count}')
print(f'Average Athletes (< 13.5 & > 11 sec): {medium_count}')
print(f'Slow athletes (> 13.5 sec): {slow_count}')
print()

print('The most stable sprinter:')
print(f'athlete #{stable_var_ind + 1} with variance {round(stable_var, 2)}')
print()

print(f'The days when everyone ran faster than 14s: {fast_days.tolist()}')
print()

print(f'Days when someone jumped > 7.5m: {long_jump_days.tolist()}')

