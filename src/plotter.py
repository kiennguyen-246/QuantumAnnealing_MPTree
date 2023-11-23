from matplotlib import pyplot as plt

# plt.xlabel("Chain Str. Pref.")
# plt.ylabel("Successful Rate (%)")
# plt.plot([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
#          [i / 100 for i in [2, 23, 80, 518, 670, 751, 341, 178, 113, 62]])

# plt.show()

plt.xlabel("Annealing Time (μs)")
plt.ylabel("Successful Rate (%)")
plt.plot([50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
         [i / 100 for i in [417, 544, 600, 602, 708, 1003, 715, 548, 413, 347]])

plt.show()