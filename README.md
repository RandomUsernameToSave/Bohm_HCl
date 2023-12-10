# HCl molecule Bomh Vizualizer !

First things first, the results are inspired by the video of minutephysics : https://www.youtube.com/watch?v=W2Xb2GFK2yc&t=130s .

## How does it work ?

We solve analytically for a harmonic potential. We then proceed to store those data, we then compute the evolution of those particles.
We then plot using the code in blender_viz.py.
Variable to have fun with : 
1. N, the grid size (NxNxN)
2. k,l,m quantum numbers that appears in the solution.
3. R0 using D(r-R0)² for potential.
4. mu

## First run : 

1. In console, run : python analytical_solution.py.
Your results are stored in pickle files. (They may be large be careful !) 
2. The open Blender, go to script and copy paste blender_viz.py in script tab. Replace, file_traj with the absolute position of the pickle file of your particles trajectories.

## Feel free to ask any question.
I'll post some changes and other uses of the code in the next week. (As you can see, we have A LOT of code not used in this simple application)