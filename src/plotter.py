from matplotlib import pyplot as plt

plt.title("Steiner Tree Found Rate for K_5 graph")
plt.xlabel("Chain Str. Pref.")
plt.ylabel("Successful Rate (%)")
plt.plot([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
         [i / 100 for i in [897, 1136, 1375, 1398, 1226, 1725, 1180, 1445, 857]],
         label="Annealing Time = 200 μs")
plt.plot([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
         [i / 100 for i in [1254, 1180, 1276, 1080, 1164, 1129, 984, 952, 762]],
         label="Annealing Time = 400 μs")
plt.legend()

plt.show()

plt.title("Optimal Solution Found Rate for K_5 graph")
plt.xlabel("Chain Str. Pref.")
plt.ylabel("Successful Rate (%)")
plt.plot([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
         [i / 100 for i in [0, 2, 3, 6, 6, 12, 7, 7, 4]],
         label="Annealing Time = 200 μs")
plt.plot([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
         [i / 100 for i in [1, 11, 4, 0, 1, 4, 11, 4, 2]],
         label="Annealing Time = 400 μs")
plt.legend()

plt.show()

# plt.xlabel("Annealing Time (μs)")
# plt.ylabel("Successful Rate (%)")
# plt.plot([50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
#          [i / 100 for i in [417, 544, 600, 602, 708, 1003, 715, 548, 413, 347]])
#
# plt.show()