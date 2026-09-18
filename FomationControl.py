import numpy as np
import matplotlib.pyplot as plt

N = 20
p = 0.25

dt = 0.02

a = 0.35
b = 0.25

pause_time = 0.03
max_steps = 1200
tolerance = 0.05

NAME = "PRATHEEK"


# --------------------------------------------------
# Generate connected Erdos-Renyi graph
# --------------------------------------------------

def connected_graph(N, p):
    while True:
        A = np.zeros((N, N))

        for i in range(N):
            for j in range(i + 1, N):
                if np.random.rand() < p:
                    A[i, j] = 1
                    A[j, i] = 1

        # Check connectivity using simple BFS
        visited = {0}
        stack = [0]

        while stack:
            i = stack.pop()

            for j in range(N):
                if A[i, j] == 1 and j not in visited:
                    visited.add(j)
                    stack.append(j)

        if len(visited) == N:
            return A


A = connected_graph(N, p)


# --------------------------------------------------
# Letter shapes
# --------------------------------------------------

def line(x1, y1, x2, y2, n):
    t = np.linspace(0, 1, n)
    return np.column_stack((
        x1 + t * (x2 - x1),
        y1 + t * (y2 - y1)
    ))


def make_letter(letter):

    if letter == "P":
        parts = [
            line(-1, -1, -1, 1, 8),
            line(-1, 1, 0.5, 1, 4),
            line(0.5, 1, 1, 0.5, 3),
            line(1, 0.5, 0.5, 0, 3),
            line(0.5, 0, -1, 0, 2)
        ]

    elif letter == "R":
        parts = [
            line(-1, -1, -1, 1, 7),
            line(-1, 1, 0.5, 1, 4),
            line(0.5, 1, 1, 0.5, 3),
            line(1, 0.5, 0.5, 0, 3),
            line(0.5, 0, -1, 0, 2),
            line(0, 0, 1, -1, 3)
        ]

    elif letter == "A":
        parts = [
            line(-1, -1, 0, 1, 7),
            line(0, 1, 1, -1, 7),
            line(-0.5, 0, 0.5, 0, 6)
        ]

    elif letter == "T":
        parts = [
            line(-1, 1, 1, 1, 10),
            line(0, 1, 0, -1, 10)
        ]

    elif letter == "H":
        parts = [
            line(-1, -1, -1, 1, 7),
            line(1, -1, 1, 1, 7),
            line(-1, 0, 1, 0, 6)
        ]

    elif letter == "E":
        parts = [
            line(1, 1, -1, 1, 5),
            line(-1, 1, -1, -1, 8),
            line(-1, 0, 0.7, 0, 4),
            line(-1, -1, 1, -1, 5)
        ]

    elif letter == "K":
        parts = [
            line(-1, -1, -1, 1, 8),
            line(-1, 0, 1, 1, 6),
            line(-1, 0, 1, -1, 6)
        ]

    else:
        raise ValueError("Letter not defined")

    points = np.vstack(parts)

    # Pick exactly N formation points
    index = np.linspace(0, len(points) - 1, N).astype(int)

    return points[index]


# --------------------------------------------------
# Initial random positions
# --------------------------------------------------

x = np.random.uniform(-5, 5, (N, 2))

# Initial formation offsets
r = make_letter(NAME[0])


# --------------------------------------------------
# Simulation
# --------------------------------------------------

for letter_number, letter in enumerate(NAME):

    r = make_letter(letter)

    for k in range(max_steps):

        u_form = np.zeros((N, 2))
        u_track = np.zeros((N, 2))

        for i in range(N):

            for j in range(N):
                if A[i, j] == 1:
                    u_form[i] += a * (
                        (x[j] - x[i]) -
                        (r[j] - r[i])
                    )

            u_track[i] = -b * (x[i] - r[i])

        u = u_form + u_track

        x = x + dt * u

        error = np.max(
            np.linalg.norm(x - r, axis=1)
        )

        if k % 3 == 0:

            plt.clf()

            # Communication graph
            for i in range(N):
                for j in range(i + 1, N):

                    if A[i, j] == 1:
                        plt.plot(
                            [x[i, 0], x[j, 0]],
                            [x[i, 1], x[j, 1]],
                            'k-',
                            alpha=0.15
                        )

            # Robots
            plt.scatter(
                x[:, 0],
                x[:, 1],
                s=45
            )

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2, 2)
            plt.gca().set_aspect('equal')

            plt.title(
                "Formation: " +
                NAME[:letter_number + 1] +
                "    |    Error: " +
                f"{error:.3f}"
            )

            plt.pause(pause_time)

        if error < tolerance:
            break

    # Hold the completed letter
    for _ in range(40):

        plt.clf()

        for i in range(N):
            for j in range(i + 1, N):

                if A[i, j] == 1:
                    plt.plot(
                        [x[i, 0], x[j, 0]],
                        [x[i, 1], x[j, 1]],
                        'k-',
                        alpha=0.15
                    )

        plt.scatter(
            x[:, 0],
            x[:, 1],
            s=45
        )

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2, 2)
        plt.gca().set_aspect('equal')

        plt.title("Letter: " + letter)

        plt.pause(0.04)


plt.show()