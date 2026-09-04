import matplotlib.pyplot as plt
import numpy as np

# Potential values (V)
E = np.array([-0.9, -0.8, -0.7, -0.6])

# Current density values (uA/cm^2)
J_MTG_ganglion = np.array([-660, -590, -500, -410])
J_PBS = np.array([-870, -770, -600, -480])

plt.figure(figsize=(7,5))

plt.plot(E, J_MTG_ganglion, marker='s', linewidth=2, markersize=7, label='Locust ganglion')
plt.plot(E, J_PBS, marker='o', linewidth=2, markersize=7, label='PBS')

plt.xlabel('(Potential vs Ag/AgCl) V', fontsize=12)
plt.ylabel('J (uA/cm2)', fontsize=12)
plt.title('Current Density of PEDOT Tip Electrodes', fontsize=14)

plt.legend(frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.xticks([-0.9, -0.8, -0.7, -0.6])

# Save the figure
save_path = r"C:\Users\simmons\Desktop\Exploring PSDs\PEDOT_current_density_comparison.png"
plt.savefig(save_path, dpi=300)

# Show the plot
plt.show()