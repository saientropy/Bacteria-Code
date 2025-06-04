# File Overview

This document gives a short summary of the purpose of each Python script in the repository.  Many files produce plots of pressure, tension or surface area and a number of them generate animations.

## Analytical
- `analytical.py` – interactive slider interface to explore cell wall parameters.
- `analytical_1.py` – alternative version of the interactive model.
- `analytical_2.py` – variant with extended options.
- `trial.py` – quick test script for the analytical model.
- `zoom.py` – helper to zoom into regions of interest.

## EVERYTHING
- `2atm.py` – plot data for 2 atmosphere pressure.
- `Animation/blender.py` – Blender script to build a 3D animation.
- `Animation/cgpt.py` – Matplotlib animation of tension and area.
- `Animation/claude.py` – alternate animation example.
- `Animation/grok.py` – demonstration of animated plotting.
- `Animation/trial.py` – small animation test.
- `area/25area.py` – generate area curve with wobble (25° takeoff).
- `area/30area.py` – area curve for 30° takeoff.
- `area/35area.py` – area curve for 35° takeoff.
- `combineplots.py` – utility to combine the outputs from multiple scripts.
- `component_contribution.py` – analyse contributions of cell wall components.
- `newdataplots/area.py` – plot area from new dataset.
- `newdataplots/pressure.py` – plot pressure from new dataset.
- `plots/area_1.py` – helper for constructing wobbly area curves.
- `plots/area_2.py` – second variant of area curve generation.
- `plots/area_3.py` – third variant of area curve generation.
- `plots/area_reference.py` – reference area curve for comparison.
- `plots/modification.py` – modifications of plot styles.
- `plots/pressure2_5.py` – pressure plot for 2.5 atm.
- `plots/pressure3_5.py` – pressure plot for 3.5 atm.
- `plots/pressure4_5.py` – pressure plot for 4.5 atm.
- `plots2/area1.py` – additional area plotting script.
- `plots2/area2.py` – additional area plotting script.
- `plots2/area3.py` – additional area plotting script.
- `pressure/2_5atm.py` – compute curves at 2.5 atm.
- `pressure/2atm.py` – compute curves at 2 atm.
- `pressure/3atm.py` – compute curves at 3 atm.
- `single shell/pg.py` – single-shell phosphatidylglycerol curve.
- `single shell/pm.py` – single-shell phosphatidylethanolamine curve.
- `strain_analysis.py` – analyse strain versus pressure using CSV data.
- `trial.py` – miscellaneous experiments.
- `vector_plots.py` – visualise component vectors.
- `vesicle_animation.py` – animate vesicle expansion in 3D.
- `vesicle_animation_2d.py` – 2D version of vesicle animation.
- `volume_pressure_analysis.py` – analyse volume change with pressure.

## all other code
- `B.py` – solve coupled equations for strain versus pressure.
- `Fea_values.py` – compute finite element analysis values.
- `diameter.py` – helper to plot wobbly diameter curves.
- `grok.py` – scratch calculations and plotting experiments.
- `o.py` – small helper function collection.
- `piecewise.py` – piecewise curve generator.
- `piecewise_curved.py` – curved variant of piecewise generator.
- `takeoff_trial.py` – experiment with takeoff angle algorithm.
- `trial.py` – generic experimental script.
- `twopointfivetakeoff.py` – compute curves at 2.5 atm takeoff.
- `yo.py` – miscellaneous utilities.
