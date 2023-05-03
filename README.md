# Game_of_Flies
AE6102 : Parallel Scientific Computing and Visualisation
Course Project

----
## Team Members: 
1. Vinit Doke (190260018)
2. Tanyut Sharma (190100128)
3. Mihir Agre (190260030)
---
## Setup
1. Clone the repository
```
git clone https://github.com/vinitdoke/Game_of_Flies.git
```
2. Install the requirements :


CUDA is required for the simulation to run.
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
---
## Documentation

| File | Link |
| --- | --- |
| Project Proposal | [Project Proposal](https://github.com/vinitdoke/Game_of_Flies/blob/cuda_3D/Documentation/AE6102_ProjectProposal_Team_Beekeepers.pdf)|
| Project Update 1 | [Update 1](https://github.com/vinitdoke/Game_of_Flies/blob/cuda_3D/Documentation/Project_Update_1.pdf) |
| Project Update 2 | [Update 2](https://github.com/vinitdoke/Game_of_Flies/blob/cuda_3D/Documentation/Project_Update_2.pdf) |
| Project Update 3 | On Moodle |
| Final Report     | [Final Report](https://github.com/vinitdoke/Game_of_Flies/blob/cuda_3D/Documentation/ae6102_project_report.pdf) |
| Final Presentation | TBD|

---
## File Structure
```
C:.
│   .gitignore
│   README.md
│
├───discussion
│   │   architechture.pdf
│   │   clusters_force_profile.pdf
│   │   Untitled-1.ipynb
│   │   
│   └───archive
│           architechture.xopp
│           clusters_force_profile.xopp
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
│   │   
│   └───outputs
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