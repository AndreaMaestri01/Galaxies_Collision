# Galaxies_Collision
Project carried out for the Astrophysics course during my master's degree.
# Galaxy Collision Simulation

This project simulates the dynamical evolution of two colliding disk galaxies using a Leapfrog integration scheme. The code was developed as part of the Astrophysics course during my master's degree in theoretical physics.

## Overview

The simulation models two galaxies represented as flattened stellar disks composed of concentric rings of stars. Each galaxy is characterized by a set of physical parameters including mass, radius, orientation, and initial position and velocity. The code computes the gravitational interaction between the stars and galactic centers over time, generating a 3D visualization of the collision.

## Features

- Galaxy initialization using user-defined physical and geometrical parameters.
- Rodrigues rotation for arbitrary disk orientation.
- Leapfrog integrator for numerically stable time evolution.
- 3D visualization of galactic disks and their dynamical interaction.
- Snapshot and image generation at selected time steps.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Astropy


