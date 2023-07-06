import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
import tarfile
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz, active_from_xyz
from SimPEG.utils import model_builder
from SimPEG import (
    maps,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
    utils,
)
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.utils.io_utils.io_utils_electromagnetics import read_dcip2d_ubc
from discretize import TensorMesh
try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver
from SimPEG.electromagnetics.static.utils.static_utils import (
    plot_pseudosection,
)
mpl.rcParams.update({"font.size": 16})


# files to work with
topo_filename =  "WN2\\topo_xyz.txt"
data_filename =  "WN2\\dc_data.obs"


# Load data
topo_xyz = np.loadtxt(str(topo_filename))
dc_data = read_dcip2d_ubc(data_filename, "volt", "surface")


# Plot voltages pseudo-section
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    dc_data,
    plot_type="scatter",
    ax=ax1,
    scale="log",
    cbar_label="V/A",
    scatter_opts={"cmap": mpl.cm.viridis},
)
ax1.set_title("Normalized Voltages")
plt.show()

# Plot apparent conductivity pseudo-section
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.78])
plot_pseudosection(
    dc_data,
    plot_type="contourf",
    ax=ax1,
    scale="log",
    data_type="apparent resistivity",
    cbar_label="S/m",
    mask_topography=True,
    contourf_opts={"levels": 20, "cmap": mpl.cm.viridis},
)
ax1.set_title("Apparent Conductivity")
plt.show()


dc_data.standard_deviation = 0.05 * np.abs(dc_data.dobs)

corexlength = 63
corezlength = 12.6
dx = 0.25
dz = 0.25
npad_x =14
npad_z = 14
pad_rate_x = 1
pad_rate_z = 1
ncx = int(np.round(corexlength / dx))
ncz = int(np.round(corezlength / dz))
hx = [(dx, npad_x, -pad_rate_x), (dx, ncx), (dx, npad_x, pad_rate_x)]
hz = [(dz, npad_z, -pad_rate_z), (dz, ncz)]
x0_mesh = -((dx * pad_rate_x ** (np.arange(npad_x) + 1)).sum() + dx * 0 - 0)
z0_mesh = -((dz * pad_rate_z ** (np.arange(npad_z) + 1)).sum() + dz * ncz) + 0
h = [hx, hz]
x0_for_mesh = [x0_mesh, z0_mesh]

mesh = TensorMesh(h, x0=x0_for_mesh)


topo_2d = np.unique(topo_xyz[:, [0, 2]], axis=0)

# Find cells that lie below surface topography
ind_active = active_from_xyz(mesh, topo_2d)

# Extract survey from data object
survey = dc_data.survey

# Shift electrodes to the surface of discretized topography
survey.drape_electrodes_on_topography(mesh, ind_active, option="top")

# Reset survey in data object
dc_data.survey = survey



air_conductivity = np.log(1e+8)
background_conductivity = np.log(30)

active_map = maps.InjectActiveCells(mesh, ind_active, np.exp(air_conductivity))
nC = int(ind_active.sum())

conductivity_map = active_map * maps.ExpMap()

# Define model
starting_conductivity_model = background_conductivity * np.ones(nC)


simulation = dc.simulation_2d.Simulation2DNodal(
    mesh, survey=survey, rhoMap=conductivity_map, solver=Solver, storeJ=True
)


dmis = data_misfit.L2DataMisfit(data=dc_data, simulation=simulation)

# Define the regularization (model objective function)
reg = regularization.WeightedLeastSquares(
    mesh,
    active_cells=ind_active,
    reference_model=starting_conductivity_model,
)

reg.reference_model_in_smooth = True  # Reference model in smoothness term

# Define how the optimization problem is solved. Here we will use an
# Inexact Gauss Newton approach.
opt = optimization.InexactGaussNewton(maxIter=10)

# Here we define the inverse problem that is to be solved
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

#

# Apply and update sensitivity weighting as the model updates
update_sensitivity_weighting = directives.UpdateSensitivityWeights()

# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)

# Set the rate of reduction in trade-off parameter (beta) each time the
# the inverse problem is solved. And set the number of Gauss-Newton iterations
# for each trade-off paramter value.
beta_schedule = directives.BetaSchedule(coolingFactor=3, coolingRate=2)

# Options for outputting recovered models and predicted data for each beta.
save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

# Setting a stopping criteria for the inversion.
target_misfit = directives.TargetMisfit(chifact=1)

# Update preconditioner
update_jacobi = directives.UpdatePreconditioner()

directives_list = [
    update_sensitivity_weighting,
    starting_beta,
    beta_schedule,
    save_iteration,
    target_misfit,
    update_jacobi,
]


dc_inversion = inversion.BaseInversion(inv_prob, directiveList=directives_list)

# Run inversion
recovered_conductivity_model = dc_inversion.run(starting_conductivity_model)
np.savetxt("WN2\inverted_model.txt", recovered_conductivity_model)
# recovered_conductivity_model = np.loadtxt("inverted_model1.txt")
norm = Normalize(vmin=0, vmax=50)


# # Plot Recovered Model
fig = plt.figure(figsize=(9, 4))

recovered_conductivity = conductivity_map * recovered_conductivity_model
recovered_conductivity[~ind_active] = np.NaN

ax1 = fig.add_axes([0.14, 0.17, 0.68, 0.7])
mesh.plot_image(
    recovered_conductivity, normal="Y", ax=ax1, grid=False, pcolor_opts={"norm": norm}
)
ax1.set_xlim(-3.5, 63.5)
ax1.set_ylim(-16, 0)
ax1.set_title("Recovered Conductivity Model")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("z (m)")

ax2 = fig.add_axes([0.84, 0.17, 0.03, 0.7])
cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical")
cbar.set_ticks([1, 10, 100, 500])
cbar.set_label(r"$\sigma$ (S/m)", rotation=270, labelpad=15, size=12)

plt.show()

