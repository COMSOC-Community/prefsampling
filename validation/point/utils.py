def add_scatter_points(points, ax, limits, title=""):
    args = [points[:, 0]]
    ax.set_xlim(limits[0][0], limits[0][1])
    if len(points[0]) > 1:
        args.append(points[:, 1])
        ax.set_ylim(limits[1][0], limits[1][1])
    else:
        args.append([0 for _ in range(len(points))])
    if len(points[0]) > 2:
        args.append(points[:, 2])
        ax.set_zlim(limits[2][0], limits[2][1])
    ax.scatter(*args, s=1)
    ax.set_title(title)
