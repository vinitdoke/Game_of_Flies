# :honeybee: Game_of_Flies
AE6102 : Parallel Scientific Computing and Visualisation
Course Project

----
## Team Members: 
1. Vinit Doke (190260018)
2. Tanyut Sharma (190100128)
3. Mihir Agre (190260030)
---
### Abstract
*Time evolution of a particle-system based on two interaction rulesets : Boids and clusters, is simulated
via numba CUDA, numba parallel, and numba serial implementations in python. Visualisation is done
via VisPy-based 2D and 3D widgets embedded inside a PyQt5-based GUI for interactivity. A pathway
to pre-compute the solution (bypassing the GUI) and visualise later, using either a live VisPy widget or
rendering to a video is provided. Parallelization of binning (both 2D and 3D) for reducing time-complexity
to O(n), support for multiple interactive-species of particles, and periodic boundary conditions are also
implemented.*
___
## Setup
1. Clone the repository
```
git clone https://github.com/vinitdoke/Game_of_Flies.git
```
2. Install the requirements :


`CUDA` is required for the simulation to run.
```
pip install -r requirements.txt
```
---
## Usage
Inside the main/ directory :
1. Run the simulation in UI
```
python hive.py -i
```
2. Run the simulation in CLI
```
python hive.py -b -r "output_directory"
```
### Demo Videos
|Video|Link|
|---|---|
|Clusters 2D| [Link](https://youtu.be/mEeR2FnSDng)|
|Clusters and Boids 3D| [Link](https://youtu.be/gRxERbZKX5M)|
|GUI|[Link](https://youtu.be/PulBwRWKz0Q) |

---
## Documentation

| File | Link |
| --- | --- |
| Project Proposal | [Project Proposal](https://github.com/vinitdoke/Game_of_Flies/blob/cuda_3D/Documentation/AE6102_ProjectProposal_Team_Beekeepers.pdf)|
| Project Update 1 | [Update 1](https://github.com/vinitdoke/Game_of_Flies/blob/cuda_3D/Documentation/Project_Update_1.pdf) |
| Project Update 2 | [Update 2](https://github.com/vinitdoke/Game_of_Flies/blob/cuda_3D/Documentation/Project_Update_2.pdf) |
| Project Update 3 | On Moodle |
| Final Report     | [Final Report](https://github.com/vinitdoke/Game_of_Flies/blob/cuda_3D/Documentation/ae6102_project_report.pdf) |
| Final Presentation | [Final Presentation](https://docs.google.com/presentation/d/1HQRkXSnLZgb5EYYDscakcldBhuxE6dnR-tViti8z1qU/edit?usp=sharing)|

---
## File Structure
```
C:.
│   .gitignore
│   README.md
│   requirements.txt
│           
├───Documentation
│   │   AE6102_ProjectProposal_Team_Beekeepers.pdf
│   │   ae6102_project_report.pdf
│   │   Project_Update_1.pdf
│   │   Project_Update_2.pdf
│   │   
│   └───discussion
│       │   architechture.pdf
│       │   clusters_force_profile.pdf
│       │   Untitled-1.ipynb
│       │   
│       └───archive
│               architechture.xopp
│               clusters_force_profile.xopp
│               
├───main
│   │   acceleration_updater.py
│   │   force_profiles.py
│   │   hive.py
│   │   icon.png
│   │   integrator.py
│   │   npy2vid.py
│   │   README.md
│   │   simulation.py
│   │   state_parameters.py
│   │   ui_container.py
│   │   vizman.py
|   └───outputs
│           
├───results_and_benchmarks
│       jit_vs_list.png
│       
└───utils
        convert_to_video.py
        mayavi_viz_check.py
        vid.npz
        vispy_anim_check.py
        vispy_check.py

```
