from keras.models import load_model
import numpy as np
import gymnasium as gym
import random
from statistics import mean, median

model = load_model('gamemodel.h5')

env = gym.make('CartPole-v1', render_mode='human') #

env.reset()
goal_steps = 3000
env._max_episode_steps = 200  # Default is 200

action = random.randrange(0, 2)  # The first action is random.
scores = []
score = 0
training_data = []
numberOfGames = 10
score_requirement = 30  # This will be incremented stepwise, to make the model stronger

for x in range(numberOfGames):
    env.reset()
    score = 0
    game_memory = []

    for t in range(goal_steps):
        observation, reward, done, truncated, info = env.step(action)  # first action is random, the rest from the model.
        prediction = model.predict(np.array([observation])).tolist()  # gets prediction from model, skal bruger næste gang.
        # print(prediction) # [[0.482.., 0.517..]] this is how the prediction looks like
        indexOfGuess = prediction[0].index(max(prediction[0])) # max() gets the largest value, index() gets its index. 
        score += reward
        if (indexOfGuess == 0):
            action = 0
            output = [1, 0]
        elif (indexOfGuess == 1):
            action = 1
            output = [0, 1]
        if done:
            scores.append(score)
            break
        game_memory.append([observation, output])
        print("Time: ", t)

    # print (game_memory)
    if score >= score_requirement:  # If a game does well, it is saved.
        for data in game_memory:  # Takes all data from game_memory and places it in training_data
            training_data.append([data[0].tolist(), data[1]])  # This list will be saved to file.

        np.save('saved.npy', np.array(training_data))  
                # NOTE: if there is nothing to save, the saved file will be destroyed.
print('Average score', mean(scores))
print('Median score', median(scores))
print('Min score', min(scores))
print('Max score', max(scores))
print('Number of training_data: ' + str(len(training_data)))

