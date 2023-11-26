Overview of files:

full_step_ns.py
- Runs a Navier Stokes full step solver with upwind discretization of the convective term. 
* This file is not designed to run in the terminal. It is a module that must be imported into other python files.
- It contains a method to calculate the fluid at the next timestep.

	calculate_step(u, v, f1, f2, rho, mu, dx, dt) 

- It also contains default tests:

	solve_ns_const_force_1(rho, mu, L, dx, dt, Niter)
	solve_ns_explosion(rho, mu, L, dx, dt, Niter, icenter, jcenter, magx, magy, radius, tstart, tend, explosion_type)

full_step_ns_tester.py
- Runs a Navier Stokes full step solver with upwind discretization of the convective term.
* Can be run in the terminal.
** Type "python3 full_step_ns_tester.py -h" for details on how to run it

half_step_ns.py
- Runs a Navier Stokes half step solver without upwind discretization of the convective term.
* This file is not designed to run in the terminal. It is a module that must be imported into other python files.

- It contains a method to calculate the fluid at the next timestep.

	calculate_step(u, v, f1, f2, rho, mu, dx, dt) 

- It also contains default tests:

	solve_ns_const_force_1(rho, mu, L, dx, dt, Niter)
	solve_ns_explosion(rho, mu, L, dx, dt, Niter, icenter, jcenter, magx, magy, radius, tstart, tend, explosion_type)

half_step_ns_tester.py
- Runs a Navier Stokes half step solver without upwind discretization of the convective term.
* Can be run in the terminal.
** Type "python3 half_step_ns_tester.py -h" for details on how to run it

half_step_upwind_ns.py
- Runs a Navier Stokes half step solver with upwind discretization of the convective term.
* This file is not designed to run in the terminal. It is a module that must be imported into other python files.

- It contains a method to calculate the fluid at the next timestep.

	calculate_step(u, v, f1, f2, rho, mu, dx, dt) 

- It also contains default tests:

	solve_ns_const_force_1(rho, mu, L, dx, dt, Niter)
	solve_ns_explosion(rho, mu, L, dx, dt, Niter, icenter, jcenter, magx, magy, radius, tstart, tend, explosion_type)

half_step_upwind_ns_tester.py
- Runs a Navier Stokes half step solver with upwind discretization of the convective term.
* Can be run in the terminal.
** Type "python3 half_step_upwind_ns_tester.py -h" for details on how to run it

half_step_upwind_ns_3D.py
- Runs a Navier Stokes half step solver with upwind discretization of the convective term in 3D.
* This file is not designed to run in the terminal. It is a module that must be imported into other python files.

- It contains a method to calculate the fluid at the next timestep.

	calculate_step(u, v, f1, f2, rho, mu, dx, dt) 

- It also contains default tests:

	solve_ns_const_force_1(rho, mu, L, dx, dt, Niter)
	solve_ns_explosion(rho, mu, L, dx, dt, Niter, icenter, jcenter, magx, magy, radius, tstart, tend, explosion_type)

test_ns.py
- Contains several tests for all three types of Navier Stokes solvers.
