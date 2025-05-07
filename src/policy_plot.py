import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

tex_fonts = {
# Use LaTeX to write all text
#     "text.usetex": True,
#     "font.family": "Times New Roman",
# Use 15pt font in plots, to match 16pt font in document
    "axes.labelsize": 20,
    "font.size": 20,
# Make the legend/label fonts a little smaller
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
}

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

plt.rcParams.update(tex_fonts)

#### this function get x, y and z and make hatchplot for x we have various evidence and y we have different costs and z is the policy
def hatchplot(x, y, z, hatches, colors, legend_labels, xlabel='X', ylabel='Y',name='name'):
    # Create a list of patches for the hatched areas
    patches = []
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):

            if z[i,j] == 1:

                rect = Rectangle((x[j], y[i]), x[j+1]-x[j], y[i+1]-y[i], hatch=hatches[1], fill=False, edgecolor=colors[1])
                patches.append(rect)
            elif z[i,j] == 0:

                rect = Rectangle((x[j], y[i]), x[j+1]-x[j], y[i+1]-y[i], hatch=hatches[0], fill=False, edgecolor=colors[0])
                patches.append(rect)

    # Create the plot
    fig, ax = plt.subplots()
    for patch in patches:
        ax.add_patch(patch)


    legend_patches = [Patch(facecolor=color, edgecolor='k') for color in colors]

    ax.legend(legend_patches, legend_labels, loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])

    plt.savefig(name+".pdf", format="pdf", bbox_inches='tight')
    plt.show()



# data for w
x = [0, 0.18, 0.38, 0.575, 0.77, 0.97, 1]
y = [0, 0.2, 0.2000000000000002, 0.2000000000000003, 0.39, 0.6, 1]
z = np.array([[1, 1, 1, 1, 1, 1],
              [0, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]])

# data for ow
zow = np.array([[1, 1, 1, 1, 1, 1],
              [1, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]])

# data for t
zt = np.array([[1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]])

# data for no evidence
znoe = np.array([[1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]])





# Define the hatch patterns and colors
hatches = ['||||', '////']
colors = ['green', 'red']
legend_labels = ['$a_k$= $E_u$', '$a_k$= $E_y$']



hatchplot(x,y,z,hatches,colors,legend_labels,xlabel='Wellbeing $w_{k-1}$', ylabel='Cost',name='wpolicy')
hatchplot(x,y,zow,hatches,colors,legend_labels,xlabel='other\'s Wellbeing $w^o_{k-1}$', ylabel='Cost',name='owpolicy')
hatchplot(x,y,zt,hatches,colors,legend_labels,xlabel='Trust $t_k$', ylabel='Cost',name='tpolicy')
hatchplot(x,y,znoe,hatches,colors,legend_labels,xlabel='No evidence', ylabel='Cost',name='nopolicy')