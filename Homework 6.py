# Value Iteration

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import copy
import itertools


# In[2]:


def track_df(textfile):
    """
    load the track txt file and save it as a numpy array
    """
    track = pd.read_csv(textfile, skiprows = 1, header = None)
    track_array = np.array(track)
    track_list_array = np.array([[j for j in track_array[i][0]] for i in range(len(track_array))])
    return track_list_array


# In[3]:


def square_position(track, s_item):    
    """
    get the square position for each type of squares
    """
    square = np.where(track == s_item)
    square_position = list(zip(square[0], square[1]))
    return square_position


# In[4]:


def update_velocity(min_velocity, max_velocity, velocity, acceleration):
    """
    update the velocity for the next state
    """
        
    updated_velocity = velocity + acceleration

    if updated_velocity <= min_velocity:
        return min_velocity
    elif updated_velocity >= max_velocity:
        return max_velocity
    else:
        return updated_velocity


# In[5]:


def crash_reset(track, crash_type, square_dict, p, wall_states_merged):
    """
    reset the car's position after crash
    """
    if crash_type == 1:
        # reset the car to the starting position
        start_pos = random.choice(square_dict['S'])
        #print('Reset to the starting position', start_pos)
        return start_pos
    elif crash_type == 2:
        # reset the car to the nearest position
        for i in range(1, max(track.shape)):
            #print('i is', i)
            options = [(p[0] + i, p[1] + i), (p[0] + i, p[1] - i), (p[0] - i, p[1] + i), (p[0] - i, p[1] - i)]
            #print('i', i)
            #print('p', p)
            #print('options', options)
            option_list = [o for o in options if o not in wall_states_merged and o[0] < track.shape[0] and o[0] >= 0 and o[1] < track.shape[1] and o[1] >= 0]
            #print('option_list', option_list)
            if len(option_list) > 0:
                #print('option list', option_list)
                closest_pos = option_list[-1]
                #print('Restart to the closet position', closest_pos)
                return closest_pos
        if len(option_list) == 0:
            return random.choice(square_dict['S'])


# In[6]:


# reference: https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
# The following algorithm used the pseudocode from wikipedia
    
def plotLineLow(x0, y0, x1, y1):
    """
    get the line path when abs(y1 - y0) < abs(x1 - x0)
    """
    
    path = []
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    
    if dy < 0:
        yi = -1
        dy = -dy
    D = (2 * dy) - dx
    y = y0
    
    for x in list(range(x0, x1 + 1)):
        path.append((x, y))
        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2 * dy
    return path


# In[7]:


def plotLineHigh(x0, y0, x1, y1):
    """
    get the line path when abs(y1 - y0) >= abs(x1 - x0)
    """
    path = []
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    
    if dx < 0:
        xi = -1
        dx = -dx
    D = (2 * dx) - dy
    x = x0
    
    for y in list(range(y0, y1 + 1)):
        path.append((x, y))
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2 * dx
    return path


# In[8]:


def breshman_algorithm(x0, y0, x1, y1):
    """   
    Bresenham algorithm 
    """
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            path = plotLineLow(x1, y1, x0, y0)
        else:
            path = plotLineLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            path = plotLineHigh(x1, y1, x0, y0)
        else:
            path = plotLineHigh(x0, y0, x1, y1)
            
    # reversed the line path so that the order of the path matches the input
    
    if (path[-1] != [x1, y1]):
        path.reverse()
    return path


# In[9]:


def state_update(t_pi_val, wall_states_merged, square_dict, a, s, status, v_list, crash_type, num_iterations, crashes):
        
    """    
    update the state
    """
    
    #print('action', a[0], a[1])
    a_x = a[0]
    a_y = a[1]
    # velocity in the x direction
    v_x = t_pi_val[s[0]][s[1]][0]
    # update the velocity with the action
    v_x_updated = update_velocity(min_velocity, max_velocity, v_x, a_x)
    # velocity in the y direction
    v_y = t_pi_val[s[0]][s[1]][1]
    # update the velocity with the action
    v_y_updated = update_velocity(min_velocity, max_velocity, v_y, a_y)
    #print('updated velocity', v_x_updated, v_y_updated)

    if status == 'success':
        # update the state to new state
        new_x = int(s[0] + v_x_updated)
        new_y = int(s[1] + v_y_updated)
        #print('new_x,y', new_x, new_y)
    elif status == 'failure':
        new_x = int(s[0] + v_x)
        new_y = int(s[1] + v_y)


    # find the path from the original state to the new state
    path = breshman_algorithm(s[0], s[1], new_x, new_y)
    #print('the path from starting position to ending position', path)
    i = 0 
    # iterate through each state in the path
    while i < len(path):
        #print('current state in the path', path[i])
        
        # if any of the state is in the wall states
        if path[i] in wall_states_merged or path[i][0] >= track.shape[0] or path[i][0] < 0 or path[i][1] >= track.shape[1] or path[i][1] < 0:
            #print('path', path[i])
            #print('Hit the wall, need to restart')
            #print('the current state that is a wall', path[i])
            # reset the car's position
            updated_coords = crash_reset(track, crash_type, square_dict, path[i], wall_states_merged)
            crashes+=1
            #print(\update the car's position\, updated_coords)
            new_x = updated_coords[0]
            #print('new position: x coordinate', new_x)
            new_y = updated_coords[1]
            #print('new position: y coordinate', new_y)
            v_x_updated = 0
            v_y_updated = 0
            break
        else:
            if path[i] in square_dict['F']:
                new_x = path[i][0]
                new_y = path[i][1]
                #print('reach the finish state', path[i])
                #print('finish line position: x coordinate, y coordinate', new_x, new_y)
                #print('Number of iterations:', num_iterations)
                break
            i = i + 1

    v_list.append([int(v_x_updated), int(v_y_updated)])
    
    #print('velocity list', v_list)
    
    a_success_x, a_success_y, a_fail_x, a_fail_y = 0, 0, 0, 0
    
    if status == 'success':
        a_success_x = new_x
        a_success_y = new_y
        #print('acceleration successfully updated, x coordinate', a_success_x)
        #print('acceleration successfully updated, y coordinate', a_success_y)
        return a_success_x, a_success_y, crashes
    elif status == 'failure':
        a_fail_x = new_x
        a_fail_y = new_y
        #print('acceleration is not successfully updated, x coordinate', a_fail_x)
        #print('acceleration is not successfully updated, y coordinate', a_fail_y)
        
        return a_fail_x, a_fail_y, crashes


# In[10]:


def q_value(square_dict, copy_t_val, a_success_x, a_success_y, a_fail_x, a_fail_y, discount_rate):
    """    
    calculate the q value
    """
    
    if (a_success_x, a_success_y) in square_dict['F']:
        return 0
    else:
        #print('copy', copy_t_val)
        #print('success', a_success_x, a_success_y, a_fail_x, a_fail_y)
        #print('Reward success', copy_t_val[a_success_x][a_success_y])
        #print('Reward failed', copy_t_val[a_fail_x][a_fail_y])
        q_val = discount_rate * (copy_t_val[a_success_x][a_success_y] * prob_success + copy_t_val[a_fail_x][a_fail_y] * prob_fail) -1
        #print('Q Value is', q_val)
        return q_val


# In[11]:


def value_iteration(track, min_velocity, max_velocity, prob_success, prob_fail, discount_rate, bellman_error, max_num_iterations, crash_type, crashes):
    
    """
    value iteration algorithm
    """
    
    num_iterations = 0
    
    # types of squares
    """
    #: wall
    S: starting line
    F: finish line
    .: open racetrack
    """
    square_list = ['#', 'S', 'F', '.']
    square_dict = {}
    for s in square_list:
        square_dict[s] = square_position(track, s)

    # actions
    actions = [-1, 0, 1]
    action_list = [[a, b] for a in actions for b in actions]
    #print('action_list', action_list)


    # table for storing pi values
    t_pi_val = np.zeros((track.shape[0], track.shape[1], 4))

    # table for storing V values
    t_v_val = np.zeros(track.shape)
    
    #print('Original V Values Table: ')
    
    #print(t_v_val)

    # identify the wall states in the pi and v values table
    for positions in square_dict['#']:
        t_pi_val[positions] = np.nan
        t_v_val[positions] = np.nan

    # get all the states except for the wall states
    non_wall_states = [value for key, value in square_dict.items() if key not in ['#', 'F']]
    non_wall_states_merged = list(itertools.chain(*non_wall_states))

    # get the wall states
    wall_states = [value for key, value in square_dict.items() if key in ['#']]
    wall_states_merged = list(itertools.chain(*wall_states))
    
    while num_iterations <= max_num_iterations:
    
        # save the original value table
        copy_t_val = copy.deepcopy(t_v_val)

        # for each states in the non-wall squares
        for s in non_wall_states_merged:
            #print('Initial State', s[0], s[1])

            v_list = []

            # pair it with each action in the list
            q_val_list = []

            for a in action_list:
                status = 'success'
                a_success_x, a_success_y, crashes = state_update(t_pi_val, wall_states_merged, square_dict, a, s, status, v_list, crash_type, num_iterations, crashes)
                #print('Updated State (action has succeeded)', (a_success_x, a_success_y))
                status = 'failure'
                a_fail_x, a_fail_y, crashes = state_update(t_pi_val, wall_states_merged, square_dict, a, s, status, v_list, crash_type, num_iterations, crashes)
                #print('Updated State (action has failed)', (a_fail_x, a_fail_y))

                q_val_list.append(q_value(square_dict, copy_t_val, a_success_x, a_success_y, a_fail_x, a_fail_y, discount_rate))
            
            #print('q_val_list', q_val_list)

            # pick the best action
            action_index = random.choice([index for index, val in enumerate(q_val_list) if val == max(q_val_list)])
            
            #print('action', action_list[action_index])
            
            t_pi_val[s[0]][s[1]] = v_list[action_index] + action_list[action_index]
            
            t_v_val[s[0]][s[1]] = max(q_val_list)
        
        num_iterations+=1
        # calculate the delta, if the delta is less than the bellman error, break
        delta = np.nanmax(abs((t_v_val - copy_t_val)))
        if delta < bellman_error:
            print('Training Complete')
            break
    
    return t_v_val, t_pi_val, crashes


# In[12]:


def test_helper(wall_states_merged, t_pi_val_updated, coord_x, coord_y, square_dict, a_x, a_y, status):
    
    """
    a helper function that checks whether the state is a wall
    """
    
    v_x = t_pi_val_updated[coord_x][coord_y][0]
    v_x_updated = update_velocity(min_velocity, max_velocity, v_x, a_x)

    v_y = t_pi_val_updated[coord_x][coord_y][1]
    v_y_updated = update_velocity(min_velocity, max_velocity, v_y, a_y)
    
    #print('initial velocity', v_x, v_y)

    if status == 'success':
        # update the state to new state
        new_x = int(coord_x + v_x_updated)
        new_y = int(coord_y + v_y_updated)
        #print('new_x,y', new_x, new_y)
    elif status == 'failure':
        new_x = int(coord_x + v_x)
        new_y = int(coord_y + v_y)

    
    # find the path from the original state to the new state
    path = breshman_algorithm(coord_x, coord_y, new_x, new_y)
    #print('the path from starting position to ending position', path)
    i = 0 
    # iterate through each state in the path
    while i < len(path):
        #print('current state in the path', path[i])
        # if any of the state is in the wall states
        if path[i] in wall_states_merged or path[i][0] >= track.shape[0] or path[i][0] < 0 or path[i][1] >= track.shape[1] or path[i][1] < 0:
            #print('path', path[i])
            #print('Hit the wall, need to restart')
            #print('the current state that is a wall', path[i])
            # reset the car's position
            updated_coords = crash_reset(track, crash_type, square_dict, path[i], wall_states_merged)
            
            #print(\update the car's position\, updated_coords)
            new_x = updated_coords[0]
            #print('new position: x coordinate', new_x)
            new_y = updated_coords[1]
            #print('new position: y coordinate', new_y)
            v_x_updated = 0
            v_y_updated = 0
            break
        else:
            if path[i] in square_dict['F']:
                new_x = path[i][0]
                new_y = path[i][1]
                break
            i = i + 1
    
    coord_x = new_x
    #print('coord_x', coord_x)
    coord_y = new_y
    #print('coord_y', coord_y)
    
    return coord_x, coord_y
    
    


# In[13]:


def test_value_iteration(track, crash_type, t_v_val_updated, t_pi_val_updated):
    
    """
    test the value iteration algorithm on track
    """
    
    square_list = ['#', 'S', 'F', '.']
    square_dict = {}
    for s in square_list:
        square_dict[s] = square_position(track, s)

    wall_states = [value for key, value in square_dict.items() if key in ['#']]
    wall_states_merged = list(itertools.chain(*wall_states))    
    start_pos = random.choice(square_dict['S'])
    coord_x = start_pos[0]
    coord_y = start_pos[1]
    
    counter = 0
    
    coords_list = []
    
    while True:
        #print('Initial State', coord_x, coord_y)
        p_state = t_pi_val_updated[coord_x][coord_y]
        a_x = int(p_state[2])
        a_y = int(p_state[3])
        #print('Current Action', a_x, a_y)
        # success
        if random.uniform(0, 1) <= 0.8:
            status = 'success'
            coord_x_updated, coord_y_updated = test_helper(wall_states_merged, t_pi_val_updated, coord_x, coord_y, square_dict, a_x, a_y, status)
            #print('Current State (Success)', coord_x_updated, coord_y_updated)
        else:
            status = 'failure'
            coord_x_updated, coord_y_updated = test_helper(wall_states_merged, t_pi_val_updated, coord_x, coord_y, square_dict, a_x, a_y, status)
            #print('Current State (Failure)', coord_x_updated, coord_y_updated)
        
        if (coord_x_updated, coord_y_updated) in square_dict['F']:
            print('Time Step', counter)
            print('Mission Complete')
            break
            
        else:
            counter += 1
            coord_x = coord_x_updated
            coord_y = coord_y_updated
            #print(coord_x, coord_y)


# In[14]:


def value_iteration_track():
    """
    apply value iteration algorithm on tracks
    """
    crashes = 0
    
    t_v_val_updated, t_pi_val_updated, crashes = value_iteration(track, min_velocity, max_velocity, prob_success, prob_fail, discount_rate, bellman_error, max_num_iterations, crash_type, crashes)
    
    print('Number of crashes', crashes)
    
    #print('V Values Table After Update:')
    
    #print(t_v_val_updated)
    
    test_value_iteration(track, crash_type, t_v_val_updated, t_pi_val_updated)
    


# In[15]:


l_track = track_df('L-track.txt')
o_track = track_df('O-track.txt')
r_track = track_df('R-track.txt')
test_track = np.array([['F', 'F', '#', '#', '#'], ['.', '.', '#', '#', '#'], ['.', '.', '#', '#', '#'], ['.', '.', '.', '.', 'S'], ['.', '.', '.', '.', 'S']])


# In[16]:


track = test_track
discount_rate = 0.9
bellman_error = 0.1
min_velocity = -5
max_velocity = 5
prob_success = 0.8
prob_fail = 0.2
max_num_iterations = 10

### Crash Type 1: Reset the car to the starting position
crash_type = 1
value_iteration_track()


# In[17]:


### Crash Type 2: Reset the car to the closest position
crash_type = 2
value_iteration_track()


# In[ ]:


track = l_track
discount_rate = 0.9
bellman_error = 0.1
min_velocity = -5
max_velocity = 5
prob_success = 0.8
prob_fail = 0.2
max_num_iterations = 10

### Crash Type 1: Reset the car to the starting position
crash_type = 1
value_iteration_track()


# In[ ]:


### Crash Type 2: Reset the car to the closest position
crash_type = 2
value_iteration_track()


# In[ ]:


track = o_track
discount_rate = 0.9
bellman_error = 0.1
min_velocity = -5
max_velocity = 5
prob_success = 0.8
prob_fail = 0.2
max_num_iterations = 10

### Crash Type 1: Reset the car to the starting position
crash_type = 1
value_iteration_track()


# In[ ]:


### Crash Type 2: Reset the car to the closest position
crash_type = 2
value_iteration_track()


# In[ ]:


track = r_track
discount_rate = 0.9
bellman_error = 0.1
min_velocity = -5
max_velocity = 5
prob_success = 0.8
prob_fail = 0.2
max_num_iterations = 10

### Crash Type 1: Reset the car to the starting position
crash_type = 1
value_iteration_track()


# In[ ]:


### Crash Type 2: Reset the car to the closest position
crash_type = 2
value_iteration_track()


# Q-learning

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import copy
import itertools


# In[2]:


def track_df(textfile):
    """
    load the track txt file and save it as a numpy array
    """
    track = pd.read_csv(textfile, skiprows = 1, header = None)
    track_array = np.array(track)
    track_list_array = np.array([[j for j in track_array[i][0]] for i in range(len(track_array))])
    return track_list_array


# In[3]:


def square_position(track, s_item):    
    """
    get the square position for each type of squares
    """
    square = np.where(track == s_item)
    square_position = list(zip(square[0], square[1]))
    return square_position


# In[4]:


def update_velocity(min_velocity, max_velocity, velocity, acceleration):
    """
    update the velocity for the next state
    """
        
    updated_velocity = velocity + acceleration

    if updated_velocity <= min_velocity:
        return min_velocity
    elif updated_velocity >= max_velocity:
        return max_velocity
    else:
        return updated_velocity


# In[5]:


def crash_reset(track, crash_type, square_dict, p, wall_states_merged):
    """
    reset the car's position after crash
    """
    if crash_type == 1:
        # reset the car to the starting position
        start_pos = random.choice(square_dict['S'])
        #print('Reset to the starting position', start_pos)
        return start_pos
    elif crash_type == 2:
        # reset the car to the nearest position
        for i in range(1, max(track.shape)):
            #print('i is', i)
            options = [(p[0] + i, p[1] + i), (p[0] + i, p[1] - i), (p[0] - i, p[1] + i), (p[0] - i, p[1] - i)]
            #print('options', options)
            option_list = [o for o in options if o not in wall_states_merged and o[0] < track.shape[0] and o[0] >= 0 and o[1] < track.shape[1] and o[1] >= 0]
            if len(option_list) > 0:
                #print('option list', option_list)
                closest_pos = random.choice(option_list)
                #print('Restart to the closet position', closest_pos)
                return closest_pos
        if len(option_list) == 0:
            return random.choice(square_dict['S'])


# In[6]:


# reference: https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
# The following algorithm used the pseudocode from wikipedia
    
def plotLineLow(x0, y0, x1, y1):
    """
    get the line path when abs(y1 - y0) < abs(x1 - x0)
    """
    
    path = []
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    
    if dy < 0:
        yi = -1
        dy = -dy
    D = (2 * dy) - dx
    y = y0
    
    for x in list(range(x0, x1 + 1)):
        path.append((x, y))
        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2 * dy
    return path


# In[7]:


def plotLineHigh(x0, y0, x1, y1):
    """
    get the line path when abs(y1 - y0) >= abs(x1 - x0)
    """
    path = []
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    
    if dx < 0:
        xi = -1
        dx = -dx
    D = (2 * dx) - dy
    x = x0
    
    for y in list(range(y0, y1 + 1)):
        path.append((x, y))
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2 * dx
    return path


# In[8]:


def breshman_algorithm(x0, y0, x1, y1):
    """   
    Bresenham algorithm 
    """
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            path = plotLineLow(x1, y1, x0, y0)
        else:
            path = plotLineLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            path = plotLineHigh(x1, y1, x0, y0)
        else:
            path = plotLineHigh(x0, y0, x1, y1)
            
    # reversed the line path so that the order of the path matches the input
    
    if (path[-1] != [x1, y1]):
        path.reverse()
    return path


# In[9]:


def state_update(t_q_val, wall_states_merged, square_dict, a_x, a_y, coords, status, crash_type, act, complete, crashes):
        
    """    
    update the state
    """
    
    s_x = coords[0]
    s_y = coords[1]
    
    v_x = t_q_val[coords][0]
    v_x_updated = update_velocity(min_velocity, max_velocity, v_x, a_x)
    v_y = t_q_val[coords][1]
    v_y_updated = update_velocity(min_velocity, max_velocity, v_y, a_y)
    #print('vx, vy', v_x, v_y)
    #print('vx, vy (updated)', v_x_updated, v_y_updated)

    if status == 'success':
        # update the state to new state
        new_x = int(s_x + v_x_updated)
        new_y = int(s_y + v_y_updated)
        #print('new_x, y', new_x, new_y)
    elif status == 'failure':
        new_x = int(s_x + v_x)
        new_y = int(s_y + v_y)
    
    # find the path from the original state to the new state
    path = breshman_algorithm(s_x, s_y, new_x, new_y)
    #print('the path from starting position to ending position', path)
    
    i = 0 
    # iterate through each state in the path
    while i < len(path):
        #print('current state in the path', path[i])
        # if any of the state is in the wall states
        if path[i] in wall_states_merged or path[i][0] >= track.shape[0] or path[i][0] < 0 or path[i][1] >= track.shape[1] or path[i][1] < 0:
            #print('path', path[i])
            #print('Hit the wall, need to restart')
            #print('the current state that is a wall', path[i])
            # reset the car's position
            updated_coords = crash_reset(track, crash_type, square_dict, path[i], wall_states_merged)
            crashes += 1
            #print("update the car's position", updated_coords)
            new_x = updated_coords[0]
            #print('new position: x coordinate', new_x)
            new_y = updated_coords[1]
            #print('new position: y coordinate', new_y)
            v_x_updated = 0
            v_y_updated = 0
            break
        else:
            if path[i] in square_dict['F']:
                new_x = path[i][0]
                new_y = path[i][1]
                #print('reach the finish state', path[i])
                #print('finish line position: x coordinate, y coordinate', new_x, new_y)
                #print('Number of iterations:', num_iterations)
                complete = True
                break
            i = i + 1

    if act == True:
        # update the current acceleration
        t_q_val[coords][2] = int(a_x)
        t_q_val[coords][3] = int(a_y)

        # update the next state's velocity
        t_q_val[new_x, new_y][0] = int(v_x_updated)
        t_q_val[new_x, new_y][1] = int(v_y_updated)


    #print('velocity list', v_list)

    a_success_x, a_success_y, a_fail_x, a_fail_y = 0, 0, 0, 0

    if status == 'success':
        a_success_x = new_x
        a_success_y = new_y
        #print('acceleration successfully updated, x coordinate', a_success_x)
        #print('acceleration successfully updated, y coordinate', a_success_y)
        return a_success_x, a_success_y, complete, crashes
    elif status == 'failure':
        a_fail_x = new_x
        a_fail_y = new_y
        #print('acceleration is not successfully updated, x coordinate', a_fail_x)
        #print('acceleration is not successfully updated, y coordinate', a_fail_y)
        return a_fail_x, a_fail_y, complete, crashes


# In[10]:


def agent_act(t_q_val, wall_states_merged, square_dict, a_x, a_y, coords, crash_type, complete, crashes, act = True):
    """
    agent takes action in state s
    """
    if random.uniform(0, 1) <= 0.8:
        status = 'success'
        a_success_x, a_success_y, complete, crashes = state_update(t_q_val, wall_states_merged, square_dict, a_x, a_y, coords, status, crash_type, act, complete, crashes)
        return a_success_x, a_success_y, complete, crashes
    else:
        status = 'failure'
        a_fail_x, a_fail_y, complete, crashes = state_update(t_q_val, wall_states_merged, square_dict, a_x, a_y, coords, status, crash_type, act, complete, crashes)
        return a_fail_x, a_fail_y, complete, crashes


# In[11]:


def optimal_action(action_list, t_q_val, wall_states_merged, square_dict, coords, status, crash_type, complete, crashes, act = False):
    
    """
    choosing the best action and return the optimal q value
    """
    
    q_val_list = []
    
    for a in action_list:
        #print('action', a)
        status = 'success'
        a_x = a[0]
        a_y = a[1]
        a_success_x, a_success_y, complete, crashes = state_update(t_q_val, wall_states_merged, square_dict, a_x, a_y, coords, status, crash_type, act, complete, crashes)
        #print(a_success_x, a_success_y)
        q_val = t_q_val[a_success_x, a_success_y, -1]
        q_val_list.append(q_val)
        #print('q list', q_val_list)
    max_q = max(q_val_list)
    #print('maximum q val', max_q)
    action_index = random.choice([index for index, val in enumerate(q_val_list) if val == max(q_val_list)])
    best_action = action_list[action_index]
    #print('best action', best_action)
    
    return max_q, best_action, complete, crashes


# In[12]:


def e_greedy(epsilon, action_list, t_q_val, wall_states_merged, square_dict, coords, crash_type, status, complete, crashes, act = False):
    
    """
    apply the epsilon-greedy policy
    """
    
    if ((1-epsilon) >= random.uniform(0, 1)):
        status = 'success'
        max_q, best_action, complete, crashes = optimal_action(action_list, t_q_val, wall_states_merged, square_dict, coords, status, crash_type, complete, crashes, act = False)
        #print('Exploitation, choose the best action from the Q table', best_action)
        return best_action
    
    else:
        #print("Exploration, choose some random action")
        random_action = random.choice(action_list)
        #print("Random Action", random_action)
        
        return random_action


# In[13]:


def q_learning(track, min_velocity, max_velocity, prob_success, prob_fail, discount_rate, bellman_error, crash_type, max_num_iterations, epsilon, lr, crashes):
    
    """
    algorithm for q learning
    """
    square_list = ['#', 'S', 'F', '.']
    square_dict = {}
    for s in square_list:
        square_dict[s] = square_position(track, s)

    # actions
    actions = [-1, 0, 1]
    action_list = [[a, b] for a in actions for b in actions]


    # table for storing q values
    # <vx, vy, ax, ay, q_val>
    t_q_val = np.zeros((track.shape[0], track.shape[1], 5))
    
    #print('Original Q Value Tables: ')
    #print(t_q_val)

    # identify the wall states in the q values table
    for positions in square_dict['#']:
        t_q_val[positions] = np.nan


    # get all the states except for the wall states
    non_wall_states = [value for key, value in square_dict.items() if key not in ['#', 'F']]
    non_wall_states_merged = list(itertools.chain(*non_wall_states))

    # get the wall states
    wall_states = [value for key, value in square_dict.items() if key in ['#']]
    wall_states_merged = list(itertools.chain(*wall_states))
    
    for counter in range(max_num_iterations):
        
        coords = random.choice(square_dict['S'])
        
        while True:
        
            complete = False

            status = 'success'

            current_action = e_greedy(epsilon, action_list, t_q_val, wall_states_merged, square_dict, coords, crash_type, status, complete, crashes, act = False)

            updated_x, updated_y, complete, crashes = agent_act(t_q_val, wall_states_merged, square_dict, current_action[0], current_action[1], coords, crash_type, complete, crashes, act = True)

            c_x = coords[0]
            c_y = coords[1]

            c_q = copy.deepcopy(t_q_val[c_x, c_y, -1])
            u_q = copy.deepcopy(t_q_val[updated_x, updated_y, -1])
            max_q, best_action, complete, crashes = optimal_action(action_list, t_q_val, wall_states_merged, square_dict, coords, status, crash_type, complete, crashes, act = False)

            if complete == True:
                reward = 0
                t_q_val[c_x, c_y, -1] = c_q + lr * (reward + discount_rate * max_q - c_q)
                #print('Training Complete')
                break
            else:
                reward = -1
                t_q_val[c_x, c_y, -1] = c_q + lr * (reward + discount_rate * max_q - c_q)
                
            coords = (updated_x, updated_y)
    
    return t_q_val, crashes


# In[14]:


def test_q_learning(track, crash_type, t_q_val, crashes):
    
    """
    test the performance of the q learning
    """
    
    square_list = ['#', 'S', 'F', '.']
    square_dict = {}
    for s in square_list:
        square_dict[s] = square_position(track, s)

    wall_states = [value for key, value in square_dict.items() if key in ['#']]
    wall_states_merged = list(itertools.chain(*wall_states))    
    start_pos = random.choice(square_dict['S'])
    coord_x = start_pos[0]
    coord_y = start_pos[1]
    
    counter = 0
    complete = False
    
    while True:
        #print('Initial State', coord_x, coord_y)
        coords = (coord_x, coord_y)
        p_state = t_q_val[coord_x][coord_y]
        a_x = int(p_state[2])
        a_y = int(p_state[3])
        #print('action', a_x, a_y)
        
        coord_x, coord_y, complete, crashes = agent_act(t_q_val, wall_states_merged, square_dict, a_x, a_y, coords, crash_type, complete, crashes, act = False)
        
        #print('Updated State', coord_x, coord_y)
        counter += 1
        #print('counter', counter)
        if (coord_x, coord_y) in [random.choice(square_dict['F'])]:
            print('Time Step', counter)
            print('Mission Complete')
            break


# In[15]:


l_track = track_df('L-track.txt')
o_track = track_df('O-track.txt')
r_track = track_df('R-track.txt')
test_track = np.array([['F', 'F', '#', '#', '#'], ['.', '.', '#', '#', '#'], ['.', '.', '#', '#', '#'], ['.', '.', '.', '.', 'S'], ['.', '.', '.', '.', 'S']])
track = test_track


# In[16]:


def q_learning_track():
    
    """
    apply Q learning algorithm on tracks
    """
    
    crashes = 0
    
    t_q_val_updated, crashes = q_learning(track, min_velocity, max_velocity, prob_success, prob_fail, discount_rate, bellman_error, crash_type, max_num_iterations, epsilon, lr, crashes)
    
    print('Number of crashes', crashes)
    
    #print('Q Values Table After Updates:')
    
    #print(t_q_val_updated)
    
    crashes = 0
    test_q_learning(track, crash_type, t_q_val_updated, crashes = 0)
    


# In[17]:


track = l_track
discount_rate = 0.9
bellman_error = 0.1
min_velocity = -5
max_velocity = 5
prob_success = 0.8
prob_fail = 0.2
max_num_iterations = 100
epsilon = 0.1
lr = 0.01
### Crash Type 1: Reset the car to the starting position
crash_type = 1
q_learning_track()


# In[18]:


### Crash Type 2: Reset the car to the closest position
crash_type = 2
q_learning_track()


# In[19]:


track = o_track
discount_rate = 0.9
bellman_error = 0.1
min_velocity = -5
max_velocity = 5
prob_success = 0.8
prob_fail = 0.2
max_num_iterations = 100
epsilon = 0.1
lr = 0.01
### Crash Type 1: Reset the car to the starting position
crash_type = 1
q_learning_track()


# In[20]:


### Crash Type 2: Reset the car to the closest position
crash_type = 2
q_learning_track()


# In[21]:


track = r_track
discount_rate = 0.9
bellman_error = 0.1
min_velocity = -5
max_velocity = 5
prob_success = 0.8
prob_fail = 0.2
max_num_iterations = 100
epsilon = 0.1
lr = 0.01
### Crash Type 1: Reset the car to the starting position
crash_type = 1
q_learning_track()


# In[22]:


### Crash Type 2: Reset the car to the closest position
crash_type = 2
q_learning_track()


# SARSA

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import copy
import itertools


# In[2]:


def track_df(textfile):
    """
    load the track txt file and save it as a numpy array
    """
    track = pd.read_csv(textfile, skiprows = 1, header = None)
    track_array = np.array(track)
    track_list_array = np.array([[j for j in track_array[i][0]] for i in range(len(track_array))])
    return track_list_array


# In[3]:


def square_position(track, s_item):    
    """
    get the square position for each type of squares
    """
    square = np.where(track == s_item)
    square_position = list(zip(square[0], square[1]))
    return square_position


# In[4]:


def update_velocity(min_velocity, max_velocity, velocity, acceleration):
    """
    update the velocity for the next state
    """
        
    updated_velocity = velocity + acceleration

    if updated_velocity <= min_velocity:
        return min_velocity
    elif updated_velocity >= max_velocity:
        return max_velocity
    else:
        return updated_velocity


# In[5]:


def crash_reset(track, crash_type, square_dict, p, wall_states_merged):
    """
    reset the car's position after crash
    """
    if crash_type == 1:
        # reset the car to the starting position
        start_pos = random.choice(square_dict['S'])
        #print('Reset to the starting position', start_pos)
        return start_pos
    elif crash_type == 2:
        # reset the car to the nearest position
        for i in range(1, max(track.shape)):
            #print('i is', i)
            options = [(p[0] + i, p[1] + i), (p[0] + i, p[1] - i), (p[0] - i, p[1] + i), (p[0] - i, p[1] - i)]
            #print('options', options)
            option_list = [o for o in options if o not in wall_states_merged and o[0] < track.shape[0] and o[0] >= 0 and o[1] < track.shape[1] and o[1] >= 0]
            if len(option_list) > 0:
                #print('option list', option_list)
                closest_pos = random.choice(option_list)
                #print('Restart to the closet position', closest_pos)
                return closest_pos
        if len(option_list) == 0:
            return random.choice(square_dict['S'])


# In[6]:


# reference: https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
# The following algorithm used the pseudocode from wikipedia
    
def plotLineLow(x0, y0, x1, y1):
    """
    get the line path when abs(y1 - y0) < abs(x1 - x0)
    """
    
    path = []
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    
    if dy < 0:
        yi = -1
        dy = -dy
    D = (2 * dy) - dx
    y = y0
    
    for x in list(range(x0, x1 + 1)):
        path.append((x, y))
        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2 * dy
    return path


# In[7]:


def plotLineHigh(x0, y0, x1, y1):
    """
    get the line path when abs(y1 - y0) >= abs(x1 - x0)
    """
    path = []
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    
    if dx < 0:
        xi = -1
        dx = -dx
    D = (2 * dx) - dy
    x = x0
    
    for y in list(range(y0, y1 + 1)):
        path.append((x, y))
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2 * dx
    return path


# In[8]:


def breshman_algorithm(x0, y0, x1, y1):
    """   
    Bresenham algorithm 
    """
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            path = plotLineLow(x1, y1, x0, y0)
        else:
            path = plotLineLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            path = plotLineHigh(x1, y1, x0, y0)
        else:
            path = plotLineHigh(x0, y0, x1, y1)
            
    # reversed the line path so that the order of the path matches the input
    
    if (path[-1] != [x1, y1]):
        path.reverse()
    return path


# In[9]:


def state_update(t_q_val, wall_states_merged, square_dict, a_x, a_y, coords, status, crash_type, act, complete, crashes):
        
    """    
    update the state
    """
    
    s_x = coords[0]
    s_y = coords[1]
    
    v_x = t_q_val[coords][0]
    v_x_updated = update_velocity(min_velocity, max_velocity, v_x, a_x)
    v_y = t_q_val[coords][1]
    v_y_updated = update_velocity(min_velocity, max_velocity, v_y, a_y)
    #print('vx, vy', v_x, v_y)
    #print('vx, vy (updated)', v_x_updated, v_y_updated)

    if status == 'success':
        # update the state to new state
        new_x = int(s_x + v_x_updated)
        new_y = int(s_y + v_y_updated)
        #print('new_x, y', new_x, new_y)
    elif status == 'failure':
        new_x = int(s_x + v_x)
        new_y = int(s_y + v_y)
    
    # find the path from the original state to the new state
    path = breshman_algorithm(s_x, s_y, new_x, new_y)
    #print('the path from starting position to ending position', path)
    
    i = 0 
    # iterate through each state in the path
    while i < len(path):
        #print('current state in the path', path[i])
        # if any of the state is in the wall states
        if path[i] in wall_states_merged or path[i][0] >= track.shape[0] or path[i][0] < 0 or path[i][1] >= track.shape[1] or path[i][1] < 0:
            #print('path', path[i])
            #print('Hit the wall, need to restart')
            #print('the current state that is a wall', path[i])
            # reset the car's position
            updated_coords = crash_reset(track, crash_type, square_dict, path[i], wall_states_merged)
            crashes+=1
            #print("update the car's position", updated_coords)
            new_x = updated_coords[0]
            #print('new position: x coordinate', new_x)
            new_y = updated_coords[1]
            #print('new position: y coordinate', new_y)
            v_x_updated = 0
            v_y_updated = 0
            break
        else:
            if path[i] in square_dict['F']:
                new_x = path[i][0]
                new_y = path[i][1]
                #print('reach the finish state', path[i])
                #print('finish line position: x coordinate, y coordinate', new_x, new_y)
                #print('Number of iterations:', num_iterations)
                complete = True
                break
            i = i + 1

    if act == True:
        # update the current acceleration
        t_q_val[coords][2] = int(a_x)
        t_q_val[coords][3] = int(a_y)

        # update the next state's velocity
        t_q_val[new_x, new_y][0] = int(v_x_updated)
        t_q_val[new_x, new_y][1] = int(v_y_updated)


    #print('velocity list', v_list)

    a_success_x, a_success_y, a_fail_x, a_fail_y = 0, 0, 0, 0

    if status == 'success':
        a_success_x = new_x
        a_success_y = new_y
        #print('acceleration successfully updated, x coordinate', a_success_x)
        #print('acceleration successfully updated, y coordinate', a_success_y)
        return a_success_x, a_success_y, complete, crashes
    elif status == 'failure':
        a_fail_x = new_x
        a_fail_y = new_y
        #print('acceleration is not successfully updated, x coordinate', a_fail_x)
        #print('acceleration is not successfully updated, y coordinate', a_fail_y)
        return a_fail_x, a_fail_y, complete, crashes


# In[10]:


def agent_act(t_q_val, wall_states_merged, square_dict, a_x, a_y, coords, crash_type, complete, crashes, act = True):
    """
    agent takes action in state s
    """
    if random.uniform(0, 1) <= 0.8:
        status = 'success'
        a_success_x, a_success_y, complete, crashes = state_update(t_q_val, wall_states_merged, square_dict, a_x, a_y, coords, status, crash_type, act, complete, crashes)
        return a_success_x, a_success_y, complete, crashes
    else:
        status = 'failure'
        a_fail_x, a_fail_y, complete, crashes = state_update(t_q_val, wall_states_merged, square_dict, a_x, a_y, coords, status, crash_type, act, complete, crashes)
        return a_fail_x, a_fail_y, complete, crashes


# In[11]:


def optimal_action(action_list, t_q_val, wall_states_merged, square_dict, coords, status, crash_type, complete, crashes, act = False):
    
    """
    choosing the best action and return the optimal q value
    """
    
    q_val_list = []
    
    for a in action_list:
        #print('action', a)
        status = 'success'
        a_x = a[0]
        a_y = a[1]
        a_success_x, a_success_y, complete, crashes = state_update(t_q_val, wall_states_merged, square_dict, a_x, a_y, coords, status, crash_type, act, complete, crashes)
        #print(a_success_x, a_success_y)
        q_val = t_q_val[a_success_x, a_success_y, -1]
        q_val_list.append(q_val)
        #print('q list', q_val_list)
    max_q = max(q_val_list)
    #print('maximum q val', max_q)
    action_index = random.choice([index for index, val in enumerate(q_val_list) if val == max(q_val_list)])
    best_action = action_list[action_index]
    #print('best action', best_action)
    
    return max_q, best_action, complete, crashes


# In[12]:


def e_greedy(epsilon, action_list, t_q_val, wall_states_merged, square_dict, coords, crash_type, status, complete, crashes, act = False):
    
    """
    apply the epsilon-greedy policy
    """
    
    if ((1-epsilon) >= random.uniform(0, 1)):
        status = 'success'
        max_q, best_action, complete, crashes = optimal_action(action_list, t_q_val, wall_states_merged, square_dict, coords, status, crash_type, complete, crashes, act = False)
        #print('Exploitation, choose the best action from the Q table', best_action)
        return best_action, crashes
    
    else:
        #print("Exploration, choose some random action")
        random_action = random.choice(action_list)
        #print("Random Action", random_action)
        
        return random_action, crashes


# In[13]:


def sarsa_learning(track, min_velocity, max_velocity, prob_success, prob_fail, discount_rate, bellman_error, crash_type, max_num_iterations, epsilon, lr, crashes):
    
    """
    algorithm for q learning
    """
    square_list = ['#', 'S', 'F', '.']
    square_dict = {}
    for s in square_list:
        square_dict[s] = square_position(track, s)

    # actions
    actions = [-1, 0, 1]
    action_list = [[a, b] for a in actions for b in actions]


    # table for storing q values
    # <vx, vy, ax, ay, q_val>
    t_q_val = np.zeros((track.shape[0], track.shape[1], 5))
    
    #print('Original Q Value Table: ')
    
    #print(t_q_val)


    # identify the wall states in the q values table
    for positions in square_dict['#']:
        t_q_val[positions] = np.nan


    # get all the states except for the wall states
    non_wall_states = [value for key, value in square_dict.items() if key not in ['#', 'F']]
    non_wall_states_merged = list(itertools.chain(*non_wall_states))

    # get the wall states
    wall_states = [value for key, value in square_dict.items() if key in ['#']]
    wall_states_merged = list(itertools.chain(*wall_states))
    
    for counter in range(max_num_iterations):
        
        curr_coords = random.choice(square_dict['S'])
        
        status = 'success'
        
        complete = False
        
        curr_actions, crashes = e_greedy(epsilon, action_list, t_q_val, wall_states_merged, square_dict, curr_coords, crash_type, status, complete, crashes, act = False)

        
        while True:
            
            u_x, u_y, complete, crashes = agent_act(t_q_val, wall_states_merged, square_dict, curr_actions[0], curr_actions[1], curr_coords, crash_type, complete, crashes, act = True)

            updated_actions, crashes = e_greedy(epsilon, action_list, t_q_val, wall_states_merged, square_dict, (u_x, u_y), crash_type, status, complete, crashes, act = False)
            
            #print('updated actions', updated_actions)
            
            c_x = curr_coords[0]
            
            c_y = curr_coords[1]
            
            c_q = copy.deepcopy(t_q_val[c_x, c_y, -1])
            
            u_q = copy.deepcopy(t_q_val[u_x, u_y, -1])
            
            #print('cx, cy, ux, uy', c_x, c_y, u_x, u_y)
   
            if complete == True:
                reward = 0
                t_q_val[c_x, c_y, -1] = c_q + lr * (reward + discount_rate * u_q - c_q)
                #print('Training Complete')
                break
            else:
                reward = -1
                t_q_val[c_x, c_y, -1] = c_q + lr * (reward + discount_rate * u_q - c_q)
                
            curr_coords = (u_x, u_y)
            
            curr_actions = updated_actions
    
    return t_q_val, crashes


# In[14]:


def test_sarsa_learning(track, crash_type, t_q_val, crashes):
    
    """
    test the performance of the q learning
    """
    
    square_list = ['#', 'S', 'F', '.']
    square_dict = {}
    for s in square_list:
        square_dict[s] = square_position(track, s)

    wall_states = [value for key, value in square_dict.items() if key in ['#']]
    wall_states_merged = list(itertools.chain(*wall_states))    
    start_pos = random.choice(square_dict['S'])
    coord_x = start_pos[0]
    coord_y = start_pos[1]
    
    counter = 0
    complete = False
    
    while True:
        #print('Initial State', coord_x, coord_y)
        coords = (coord_x, coord_y)
        p_state = t_q_val[coord_x][coord_y]
        a_x = int(p_state[2])
        a_y = int(p_state[3])
        #print('action', a_x, a_y)
        
        coord_x, coord_y, complete, crashes = agent_act(t_q_val, wall_states_merged, square_dict, a_x, a_y, coords, crash_type, complete, crashes, act = False)
        
        #print('Updated State', coord_x, coord_y)
        counter += 1
        #print('counter', counter)
        if (coord_x, coord_y) in [random.choice(square_dict['F'])]:
            print('Time Step', counter)
            print('Mission Complete')
            break


# In[15]:


l_track = track_df('L-track.txt')
o_track = track_df('O-track.txt')
r_track = track_df('R-track.txt')
test_track = np.array([['F', 'F', '#', '#', '#'], ['.', '.', '#', '#', '#'], ['.', '.', '#', '#', '#'], ['.', '.', '.', '.', 'S'], ['.', '.', '.', '.', 'S']])
track = test_track


# In[16]:


def sarsa_track():
    """
    apply sarsa algorithm on tracks
    """
    crashes = 0
    
    t_q_val_updated, crashes = sarsa_learning(track, min_velocity, max_velocity, prob_success, prob_fail, discount_rate, bellman_error, crash_type, max_num_iterations, epsilon, lr, crashes)
    
    print('Number of crashes', crashes)
    
    #print('Q Value Table After Updates: ')
    
    #print(t_q_val_updated)
    
    test_sarsa_learning(track, crash_type, t_q_val_updated, crashes)


# In[17]:


track = l_track
min_velocity = -5
max_velocity = 5
prob_success = 0.8
prob_fail = 0.2
discount_rate = 0.9
bellman_error = 0.1
max_num_iterations = 100
epsilon = 0.1
lr = 0.01
### Crash Type 1: Reset the car to the starting position
crash_type = 1
sarsa_track()


# In[18]:


### Crash Type 2: Reset the car to the closest position
crash_type = 2
sarsa_track()


# In[19]:


track = o_track
min_velocity = -5
max_velocity = 5
prob_success = 0.8
prob_fail = 0.2
discount_rate = 0.9
bellman_error = 0.1
max_num_iterations = 100
epsilon = 0.1
lr = 0.01
### Crash Type 1: Reset the car to the starting position
crash_type = 1
sarsa_track()


# In[20]:


### Crash Type 2: Reset the car to the closest position
crash_type = 2
sarsa_track()


# In[21]:


track = r_track
min_velocity = -5
max_velocity = 5
prob_success = 0.8
prob_fail = 0.2
discount_rate = 0.9
bellman_error = 0.1
max_num_iterations = 100
epsilon = 0.1
lr = 0.01
### Crash Type 1: Reset the car to the starting position
crash_type = 1
sarsa_track()


# In[22]:


### Crash Type 2: Reset the car to the closest position
crash_type = 2
sarsa_track()
