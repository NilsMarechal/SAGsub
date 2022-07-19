# SAGsub
Scripts for Streptavidin lattice subtraction

This is a beta script. It intend to be improved for precision, statistics extraction, and speed.

If used in your works, please aknowlege Nils Marechal, Ph.D, for the developement of this script.

Prior to execute, one need to edit the apix and Planes variable in SAGsub_2_fail-pass.py

- apix is the calibrated pixel size in your dataset.

- Planes is the plane spacing (in pixel) between two bragg spots in the power spectrum.

To determine Planes, simply open a good, motion corrected micrograph in imod, display powerspectum, and click on two adjacent peaks to get X and Y     coordinates. Then, calculate the 2D coordinates using the following formula:

Planes = sqrt( (X1-X2)² + (Y1-Y2)²) ; round it to first decimal.

To execute, simply go to the directory containing motion-corrected micrographs, then >python3 path-to-script/SAGsub_2_fail-pass.py
