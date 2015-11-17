                                 MCF Curve Skeletonizer

# What is it
This code implements the Mean Curvature Flow (MCF) curve skeletonizer from the 
paper "Mean Curvature Skeletons" (Computer Graphics Forum 2012, Proc. of the 
Symposium on Geometry Processing). The code is adapted from my research code 
base. It is intentionnaly kept as simple as possible, with few dependencies, in
order to be easily used in other projects. As such, only a command line demo
application is available.

# Configuration
The code has been only tested on GNU/Linux platforms, compiled with gcc 5.2. 
You need to have c++ 14 support for now, but I may get rid of this requirement
as I do not really use it in this adapted version.

# Installation
First check out this repository
    git clone https://github.com/tdelame/mcf_curve_skeletonizer.git

Then edit the file _Makefile_ to define some variables for the compilation. 

# Usage
TODO

# Licensing
This code is release under the MIT Licence. Please see the file called LICENCE.

# Contacts
Thomas Delame, tdelame@gmail.com

# Credits and acknowledgements
I would like to thanks Andrea Tagliasacchi and Ibraheem Alhashim for releasing
their code on github (https://github.com/ataiya/starlab-mcfskel.git). I want to
thank again Andrea Tagliasacchi for his constructive comments about the method.
