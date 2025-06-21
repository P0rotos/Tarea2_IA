import matplotlib.pyplot as plt

# Initialize lists to store episodes and rewards
episodes = []
rewards = []

# Open the file and read the data
with open("forplot.txt", "r") as file:
    for line in file:
        # Split the line into episode and reward
        episode, reward = map(float, line.split())
        episodes.append(episode)
        rewards.append(reward)

# Plotting the learning curve
plt.plot(episodes, rewards, 'o', markersize=1) #, marker='o'
plt.title("Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.grid(True)
plt.show()