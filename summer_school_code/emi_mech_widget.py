


import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path

class EMI_mech_widget:

    def __init__(self):

        self.directions = ["fiber", "sheet"] #, "normal"]
        self.subdomains = [0, 1]         # extra-, intracellular subdomains
        #self.time = np.linspace(0,500,500)

        input_folder = "output_calcium_matrix_stiffness_simulations"
        ca_scaling_factors = np.arange(0.5, 1.55, 0.05)
        ce_scaling_factors = np.array([0.06250, 0.08839, 0.12500, 0.17678, 0.25000, 0.35355, 0.50000, 0.70711, 1.00000, 1.41421, 2.00000, 2.82843, 4.00000, 5.65685, 8.00000, 11.31371, 16.00000])   # 2**np.arange(-4, 4, 0.5)


        self.read_input_data(input_folder, ca_scaling_factors, ce_scaling_factors)
        self.init_plot()

    def read_input_data(self, input_folder, ca_scaling_factors, ce_scaling_factors):
        data = defaultdict(lambda: defaultdict(dict))

        for ce in ce_scaling_factors:
            for ca in ca_scaling_factors:
                key_ce = round(ce, 5)
                key_ca = round(ca, 2)
                fin = Path(f"{input_folder}/output_ce_{key_ce}_ca_{key_ca}.npy")
                try:
                    d = np.load(fin, allow_pickle=True).item()
                    data[key_ce][key_ca] = d
                except FileNotFoundError:
                    print(f"Unable to find file {đin}.")

        self.time = d["time"]

        self.output_values = data

    def init_plot(self):

        directions, subdomains, output_values = self.directions, self.subdomains, self.output_values
        time = self.time

        fig, axes = plt.subplots(4, 2, figsize=(8, 8), sharex=True, sharey="row")

        legends_strain = [r"$\overline{E_{ff}}$", r"$\overline{E_{ss}}$"] #, r"$\overline{E_{nn}}$"]
        legends_stress = [r"$\overline{\sigma_{ff}}$", r"$\overline{\sigma_{ss}}$"] #, r"$\overline{\sigma_{nn}}$"]

        axes[0][0].set_title(f"Extracellular subdomain ($\Omega_e$)")
        axes[0][1].set_title(f"Intracellular subdomain ($\Omega_i$)")

        axes[-1][0].set_xlabel("Time (ms)")
        axes[-1][1].set_xlabel("Time (ms)")
        
        axes[0][0].set_ylabel("Aktive tension (kPa)")
        axes[1][0].set_ylabel("[Ca] (M)")
        axes[2][0].set_ylabel("Strain (-)")
        axes[3][0].set_ylabel("Stress (kPa)")

        axes[0][0].set_ylim(-0.1, 2.5)
        axes[1][0].set_ylim(-1.0, 500)
        axes[2][0].set_ylim(-0.25, 0.25)
        axes[3][0].set_ylim(-8.0, 8.0)

        key_ce = 1.0
        key_ca = 1.0
        
        calcium_values = output_values[key_ce][key_ca]["calcium_values"]
        active_values = output_values[key_ce][key_ca]["active_values"]
        line_ca = axes[0][1].plot(time, calcium_values)[0]
        line_active = axes[1][1].plot(time, active_values)[0]

        for dir_id, direction in enumerate(directions):
            for subdomain_id in subdomains:
                strain_values = output_values[key_ce][key_ca]["strain"][direction][subdomain_id]
                stress_values = output_values[key_ce][key_ca]["stress"][direction][subdomain_id]
           
                line_strain = axes[2][subdomain_id].plot(time, strain_values)[0]
                line_stress = axes[3][subdomain_id].plot(time, stress_values)[0]
        
        axes[2][1].legend(legends_strain)
        axes[3][1].legend(legends_stress)

        for ax_l in axes:
            for ax in ax_l:
                ax.axvline(x=0, color="gray", linewidth=1)
                ax.axhline(y=0, color="gray", linewidth=0.5)

        plt.tight_layout()

        self.fig, self.axes, self.output_values = fig, axes, output_values

    def update_plot(self, stiffness_scale, ca_scale):
        fig, axes, output_values = self.fig, self.axes, self.output_values
        directions, subdomains = self.directions, self.subdomains

        key_ce = round(stiffness_scale, 5)
        key_ca = round(ca_scale, 2)
        
        for subdomain_id in subdomains:
            calcium_values = output_values[key_ce][key_ca]["calcium_values"]
            active_values = output_values[key_ce][key_ca]["active_values"]
            
            axes[0][1].lines[0].set_ydata(calcium_values)
            axes[1][1].lines[0].set_ydata(active_values)

        for dir_id, direction in enumerate(directions):
            for subdomain_id in subdomains:
                strain_values = output_values[key_ce][key_ca]["strain"][direction][subdomain_id]
                stress_values = output_values[key_ce][key_ca]["stress"][direction][subdomain_id]

                axes[2][subdomain_id].lines[dir_id].set_ydata(strain_values)
                axes[3][subdomain_id].lines[dir_id].set_ydata(stress_values)

        fig.canvas.draw()

