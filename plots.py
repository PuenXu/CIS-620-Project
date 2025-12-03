import matplotlib.pyplot as plt

# ----------------------------------------
# TEMPLATE INPUTS (FILL THESE IN)
# ----------------------------------------

# Plot 1 (6 x-values)
x_col1 = range(6) # length 6
y11_lineA = [174, 206, 258, 1200, 1200, 1200]                # plot (1,1)
y11_lineB = [344, 214, 391, 293, 1200, 1200]
y21_lineA = [1, 1, 1, 0.3188, 0.2811, 0.2708]                # plot (2,1)
y21_lineB = [1, 1, 1, 1, 0.4251, 0.3005]

# Plot 2 (10 x-values)
x_col2 = range(10)  # length 10
y12_lineA = [98, 119, 105, 130, 223, 1200, 1200, 1200, 1200, 1200]
y12_lineB = [89, 108, 95, 127, 121, 203, 1200, 1200, 1200, 1200]
y22_lineA = [1, 1, 1, 1, 1, 0.8707, 0.3146, 0.276, 0.284, 0.258]
y22_lineB = [1, 1, 1, 1, 1, 1, 0.4573, 0.793, 0.313, 0.302]

# Plot 3 (13 x-values)
x_col3 = range(13)  # length 13
y13_lineA = [175, 262, 209, 435, 301, 499, 1200, 1200, 1200, 1200, 1200, 1200, 1200]
y13_lineB = [181, 213, 283, 208, 276, 221, 440, 511, 1200, 1200, 1200, 1200, 1200]
y23_lineA = [1, 1, 1, 1, 1, 1, 0.9005, 0.3249, 0.145, 0.227, 0.14, 0.136, 0.18]
y23_lineB = [1, 1, 1, 1, 1, 1, 1, 1, 0.9001, 0.424, 0.672, 0.245, 0.19]

# Names for plots and lines
plot_titles = [
    ["Small Map", "Medium Map", "Large Map"],   # Row 1
    ["Small Map", "Medium Map", "Large Map"]   # Row 2
]
line_labels = ["Learning", "No Learning"]

y_labels = ["# of steps executed", "Completion ratio"]
x_label = "# of malicious robots"

# ----------------------------------------
# PLOTTING
# ----------------------------------------

x_sets = [x_col1, x_col2, x_col3]

# Y-values organized for looping
y_sets = [
    [ (y11_lineA, y11_lineB), (y12_lineA, y12_lineB), (y13_lineA, y13_lineB) ],  # row 1
    [ (y21_lineA, y21_lineB), (y22_lineA, y22_lineB), (y23_lineA, y23_lineB) ]   # row 2
]


# ---------------------------------------------------
# PLOTTING
# ---------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 8))

for row in range(2):
    for col in range(3):
        ax = axes[row, col]

        # Get x-values for this column
        xvals = x_sets[col]

        # Plot both lines
        yA, yB = y_sets[row][col]
        ax.plot(xvals, yA, label=line_labels[0])
        ax.plot(xvals, yB, label=line_labels[1])

        # Title
        ax.set_title(plot_titles[row][col])

        # Labels
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_labels[row])

        # Legend
        ax.legend()

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()