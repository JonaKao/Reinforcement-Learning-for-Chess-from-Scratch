import matplotlib.pyplot as plt
import csv

timesteps, rewards = [], []
with open("models/metrics.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        timesteps.append(int(row["timesteps"]))
        rewards.append(float(row["ep_rew_mean"]))

plt.figure()
plt.plot(timesteps, rewards)
plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Reward")
plt.title("Chess Agent Learning Curve")
plt.show()
