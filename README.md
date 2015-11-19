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
Select a build type by typing one of the two following lines: 
    make release
    make debug

Once the build type is active, you can build the command line application by
typing ``make``.
    
# Usage
Assuming you have not changed the target directory, to access the command line
help, simply type:
    ./bin/mcf_curve_skeletonizer --help

For example, to perform two iterations of the algorithm on a mesh described by
the file eight.off, with the default energy weights, type:
    ./bin/mcf_curve_skeletonizer -i eight.off -o eight.graph --iterations 2

You can visualize the produced skeletons (eight.graph in the previous command)
with a 3D Skeleton Web renderer, available here on my website at the following
address http://tdelame.co.nf/pages/projects/skeleton_renderer.html

# Licensing
This code is release under the MIT Licence. Please see the file called LICENCE.

# Contacts
Thomas Delame, tdelame@gmail.com

# Credits and acknowledgements
I would like to thanks Andrea Tagliasacchi and Ibraheem Alhashim for releasing
their code on github (https://github.com/ataiya/starlab-mcfskel.git). I want to
thank again Andrea Tagliasacchi for his constructive comments about the method.
