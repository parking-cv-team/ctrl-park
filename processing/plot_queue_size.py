import pandas as pd
import matplotlib.pyplot as plt

fn = "queue_size.log"

df = pd.read_csv(fn, names=["ts", "uri", "queue_size"], header=None)
df["ts"] = df["ts"].astype(float)
df["t_rel"] = df["ts"] - df["ts"].iloc[0]

plt.plot(df["t_rel"], df["queue_size"], marker=".", linestyle="-")
plt.title("Queue size vs time (s)")
plt.xlabel("Seconds since start")
plt.ylabel("Queue size")
plt.grid(True)
plt.tight_layout()
plt.show()