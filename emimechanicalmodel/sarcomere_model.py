"""

Åshild Telle / University of Washington / 2024

"""

import dolfin as df
from mpi4py import MPI
import numpy as np

from emimechanicalmodel.cardiac_model import CardiacModel
from emimechanicalmodel.sarcomerematerial import EMIHolzapfelMaterial_with_substructures as MaterialModel
from emimechanicalmodel.sarcomerematerial import assign_discrete_values
from emimechanicalmodel.compressibility import IncompressibleMaterial, SarcomereNearlyIncompressibleMaterial
from emimechanicalmodel.proj_fun import ProjectionFunction


class SarcomereModel(CardiacModel):
    """

    Module for our EMI model extended with sarcomere structure.

    Note: One of fix_x_bnd, fix_y_bnd and fix_middle must be true.
    No more than one of these can be true.

    Args:
        mesh (df.Mesh): Domain to be used
        experiment (str): Which experiment - "contraction", "stretch_ff", "shear_fs", ...
        material_properties: parameters to underlying material model,
            default empty dictionary which means default values will be used
        verbose (int): Set to 0 (no verbose output; default), 1 (some),
            or 2 (quite a bit)
    """

    def __init__(
        self,
        mesh,
        volumes,
        sarcomere_angles,
        experiment,
        material_model="holzapfel",
        material_parameters={},
        active_model="active_strain",
        compressibility_model="incompressible",
        compressibility_parameters={},
        verbose=0,
        robin_bcs_value=0,
        fraction_sarcomeres_disabled=0.0,
        isometric=False,
    ):
        # mesh properties, subdomains
        self.verbose = verbose
        self.volumes = volumes
        self.set_subdomains(volumes)
        self.sarcomere_angles = sarcomere_angles

        U = df.FunctionSpace(mesh, "DG", 0)
        subdomain_map = volumes.array()  # only works for DG-0
        #self.xi_sarcomeres = df.Function(U)
        #assign_discrete_values(self.xi_sarcomeres, subdomain_map, 1)      # contractile units!

        mat_model = MaterialModel(U, subdomain_map, self.subdomains, **material_parameters)

        if compressibility_model=="incompressible":
            comp_model = IncompressibleMaterial()
        elif compressibility_model=="nearly_incompressible":
            comp_model = SarcomereNearlyIncompressibleMaterial(U, subdomain_map, **compressibility_parameters)
        else:
            print("Error: Unknown material model; please specify as 'incompressible' or 'nearly_incompressible'.")

        self.fraction_sarcomeres_disabled = fraction_sarcomeres_disabled

        self.U, self.subdomain_map, self.mat_model, self.comp_model = \
                U, subdomain_map, mat_model, comp_model

        super().__init__(
            mesh,
            experiment,
            active_model,
            compressibility_model,
            verbose,
            robin_bcs_value,
            isometric,
        )
   
    def _set_direction_vectors(self):

        sarcomere_angles = self.sarcomere_angles  # array-like
        subdomains = self.subdomain_map           # assumed to match self.volumes
        mesh = self.mesh

        V = df.VectorFunctionSpace(mesh, "DG", 0)
        Q = df.FunctionSpace(mesh, "DG", 0)

        fiber_dir = df.Function(V, name="Fiber direction")
        sheet_dir = df.Function(V, name="Sheet direction")
        angle_fn = df.Function(Q, name="Angle distribution (radians)")

        # 1. Build angle_fn.vector() in a parallel-safe way
        angle_vals = angle_fn.vector().get_local()
        cell_domains = self.volumes.array()  # per-cell subdomain ID

        for i in range(len(angle_vals)):  # these are local cells only
            s = cell_domains[i]
            if 1000 <= s < 2000:
                angle_vals[i] = np.pi/2 - sarcomere_angles[int(s)-1000]
            else:
                angle_vals[i] = 0.0

        angle_fn.vector().set_local(angle_vals)
        angle_fn.vector().apply("insert")  # ensure MPI sync

        # 2. Get local DOF range for the vector fields
        dofmap_V = V.dofmap()
        local_range = fiber_dir.vector().local_range()
        lo, hi = local_range
        size = hi - lo

        fiber_dir_vals = np.zeros(size)
        sheet_dir_vals = np.zeros(size)

        # Get angle array for local cells
        angle_array = angle_fn.vector().get_local()

        for cell in df.cells(mesh):
            cell_idx = cell.index()
            cell_dofs = dofmap_V.cell_dofs(cell_idx)

            # Skip if this cell's DOFs are not owned by this process
            if cell_dofs[0] < lo or cell_dofs[1] >= hi:
                continue

            theta = angle_array[cell_idx]
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            R = np.array([[cos_t, -sin_t],
                          [sin_t,  cos_t]])

            v_fib = R @ np.array([1.0, 0.0])
            v_sheet = R @ np.array([0.0, 1.0])

            fiber_dir_vals[cell_dofs[0] - lo] = v_fib[0]
            fiber_dir_vals[cell_dofs[1] - lo] = v_fib[1]

            sheet_dir_vals[cell_dofs[0] - lo] = v_sheet[0]
            sheet_dir_vals[cell_dofs[1] - lo] = v_sheet[1]

        # 3. Safely assign local values and apply
        fiber_dir.vector().set_local(fiber_dir_vals)
        sheet_dir.vector().set_local(sheet_dir_vals)

        fiber_dir.vector().apply("insert")
        sheet_dir.vector().apply("insert")

        # 4. Save fields
        self.fiber_dir = fiber_dir
        self.sheet_dir = sheet_dir

    def set_subdomains(self, volumes):
        mpi_comm = MPI.COMM_WORLD
        rank = mpi_comm.Get_rank()
        
        local_subdomains = set(volumes.array())
        subdomains = mpi_comm.gather(local_subdomains, root=0)

        if rank == 0:
            global_subdomains = []
            for s in subdomains:
                global_subdomains += s
            global_subdomains = list(set(global_subdomains))
        else:
            global_subdomains = None

        self.subdomains = mpi_comm.bcast(global_subdomains, root=0)
        self.num_subdomains = len(self.subdomains) 
        self.intracellular_space = self.subdomains[:]
        #self.intracellular_space.remove(0)       # remove matrix space
         
        # each sub-region:
        self.sarcomere_regions = filter(lambda x: 1000 <= x < 2000, local_subdomains)
        self.zline_regions = filter(lambda x: 2000 <= x < 3000, local_subdomains)
        self.cytoskeleton_regions = filter(lambda x: 3000 <= x < 4000, local_subdomains)
        self.connection_regions = filter(lambda x: 4000 <= x < 5000, local_subdomains)
        self.local_subdomains = local_subdomains

        if self.verbose == 2:
            print(f"Local subdomains (rank {rank}):{local_subdomains}")  
            print(f"Global subdomains (rank {rank}):{subdomains}")
            print("Number of subdomains in total: ", self.num_subdomains)


    def _define_active_fn(self):
        """
        Defines an active strain/stress function for active contraction.
        This function works in parallel by assigning values only to locally owned DOFs.
        """
        comm = self.U.mesh().mpi_comm()
        rank = comm.Get_rank()

        # Create function in the continuous space U
        self.active_fn = df.Function(self.U, name="Active tension")
        self.active_fn.vector().zero()

        # Cell-wise constant DG0 space
        V0 = df.FunctionSpace(self.U.mesh(), "DG", 0)
        active_dg0 = df.Function(V0)

        # Map from cell index to subdomain ID
        cell_to_subdomain = self.volumes.array()

        # Assign values to the DG0 function on locally owned cells
        for cell in df.cells(self.U.mesh()):  # Iterates over local cells only
            cell_index = cell.index()
            subdomain_id = cell_to_subdomain[cell_index]

            # Seed RNG with subdomain_id for deterministic results
            np.random.seed(subdomain_id)
            if 1000 <= subdomain_id < 2000 and np.random.uniform() >= self.fraction_sarcomeres_disabled:
                scaling_value = max(0.0, np.random.normal(1.0, 0.1))
            else:
                scaling_value = 0.0

            # Set value directly to local DOF
            dof = V0.dofmap().cell_dofs(cell_index)[0]
            active_dg0.vector()[dof] = scaling_value

        # Apply changes to ensure global consistency
        active_dg0.vector().apply("insert")

        # Interpolate to the continuous function space
        self.active_fn = df.interpolate(active_dg0, self.U)

        # Store sarcomere scaling field for reuse
        self.sarcomere_scaling = df.Function(self.U, name="Sarcomere scaling")
        self.sarcomere_scaling.assign(self.active_fn)


    def update_active_fn(self, value):
        """
        Updates the active strain/stress function by scaling the stored sarcomere field.

        Args:
            value (float): Scalar multiplier for sarcomere activity.
        """
        self.active_fn.vector().zero()
        self.active_fn.vector().axpy(value, self.sarcomere_scaling.vector())
        self.active_fn.vector().apply("insert")


    def _define_projections(self):
        """

        Defines projection objects which tracks different variables of
        interest as CG functions, defined as scalars, vectors, or tensors.

        If project is set to true in the solve call, these will be updated,
        and (for efficiency) not otherwise.

        """

        mesh = self.mesh

        # define function spaces

        U_DG = df.FunctionSpace(mesh, "DG", 1)
        V_DG = df.VectorFunctionSpace(mesh, "DG", 2)
        T_DG = df.TensorFunctionSpace(mesh, "DG", 2)

        p_DG = df.Function(U_DG, name="Hydrostatic pressure (kPa))")
        Ta_DG = df.Function(U_DG, name="Active tension (kPa)")
        u_DG = df.Function(V_DG, name="Displacement (µm)")
        E_DG = df.Function(T_DG, name="Strain")
        sigma_DG = df.Function(T_DG, name="Cauchy stress (kPa)")
        P_DG = df.Function(T_DG, name="Piola-Kirchhoff stress (kPa)")

        p_proj = ProjectionFunction(self.p, p_DG)
        Ta_proj = ProjectionFunction(self.active_fn, Ta_DG)
        u_proj = ProjectionFunction(self.u, u_DG)
        E_proj = ProjectionFunction(self.E, E_DG)
        sigma_proj = ProjectionFunction(self.sigma, sigma_DG)
        P_proj = ProjectionFunction(self.P, P_DG)

        self.u_DG = u_DG
        self.p_DG = p_DG
        self.Ta_DG = Ta_DG
        self.E_DG = E_DG
        self.sigma_DG = sigma_DG
        self.PiolaKirchhoff_DG = P_DG

        self.tracked_variables = [u_DG, p_DG, Ta_DG, E_DG, sigma_DG, P_DG]
        self.projections = [u_proj, p_proj, Ta_proj, E_proj, sigma_proj, P_proj]


