# softquake

Simulates the movement of softbody buildings under the effect of earthquakes.

Softbody structure is assumed to:
* Have Hookean springs and dampeners connecting the set of nodes.
* Have no self collision property.
* Have no effect on the earthquake plate.
* Be under the influence of gravity.

Earthquake plate is assumed to:
* Be constrained to horizontal motion.
* Move based on a singular sine with specific frequency and amplitude.
* Have no frictional interaction with the softbody.

Simulation is implemented to:
* Use a verlet integrator scheme.
* Use real distance and time units.
