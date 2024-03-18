from prefsampling.point import ball_uniform

import matplotlib.pyplot as plt

width = [4, 10]
points = ball_uniform(10000, 2, widths=width, center_point=[4, 10])
print(points)
x = points[:, 0]
y = points[:, 1]
fig = plt.figure(figsize=(width[0] * 4, width[1] * 4))
ax = fig.add_subplot(111)
ax.scatter(x, y)
plt.xlim(-width[0] - 1, width[0] + 1)
plt.ylim(-width[1] - 1, width[1] + 1)
plt.show()
