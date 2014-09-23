
# ************************************************************************
# File:    H_atom.py
# Purpose: Solve the ground-state of H atom in Hartree-Fock model, including energy correction step
# Version: FEniCS 1.4.0
# Author:  Houdong Hu
# ************************************************************************

from dolfin import *
from math import *
import numpy as np
import datetime

mesh = BoxMesh(-40,-40,-40,40,40,40,2,2,2)
origin = Point(.0,.0,.0)

# construct priori adaptive mesh

for j in range(50):
    cell_markers = CellFunction("bool",mesh)
    cell_markers.set_all(False)
    cmin = 100
    for cell in cells(mesh):
        p = cell.midpoint()
        r =((p[0]-origin[0])**2+(p[1]-origin[1])**2+(p[2]-origin[2])**2)**0.5
        if r<cmin:
           cmin=r
    for cell in cells(mesh):
        p = cell.midpoint()
        r =((p[0]-origin[0])**2+(p[1]-origin[1])**2+(p[2]-origin[2])**2)**0.5
        if r<=cmin:
            cell_markers[cell]=True
    mesh = refine(mesh, cell_markers)

for k in range(4):
    cell_markers = CellFunction("bool",mesh)
    cell_markers.set_all(True)
    mesh = refine(mesh, cell_markers)

cmin = 100
for cell in cells(mesh):
    p = cell.midpoint()
    r =((p[0]-origin[0])**2+(p[1]-origin[1])**2+(p[2]-origin[2])**2)**0.5
    if r<cmin:
       cmin=r
print cmin
print mesh.coordinates().shape

V = FunctionSpace(mesh, "Lagrange", 1)

def boundary(x, on_boundary):
    return on_boundary

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

delta=cmin
v_ext_corr = Expression("1.0/(pow(pow(x[0]-0.0,2)+pow(x[1]-0.0, 2)+pow(x[2]-0.0,2), 0.5)+delta)",delta=delta)


# Define test function and initial condition (STO-3G)
u = TrialFunction(V)
v = TestFunction(V)

# sto-3g
alpha11=3.42525091
alpha12=0.62391373
alpha13=0.16885540
tmp11=Expression("(pow(2*alpha11/pi,3.0/4.0))*exp(-alpha11*(pow(x[0]-0.0,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha11=alpha11)
tmp12=Expression("(pow(2*alpha12/pi,3.0/4.0))*exp(-alpha12*(pow(x[0]-0.0,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha12=alpha12)
tmp13=Expression("(pow(2*alpha13/pi,3.0/4.0))*exp(-alpha13*(pow(x[0]-0.0,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha13=alpha13)
rx1s=0.15432897*(interpolate(tmp11,V)).vector().array()+0.53532814*(interpolate(tmp12,V)).vector().array()+0.44463454*(interpolate(tmp13,V)).vector().array()


r1s=-0.45
r1s_p=0.0

tmp1=Function(V)
tmp1.vector()[:]=rx1s
frx1s=Function(V)
v1s=Function(V)

while True:

    time0=datetime.datetime.now()

    # normalization
    nor1=assemble(inner(tmp1,tmp1)*dx)**0.5
    frx1s.vector()[:]=tmp1.vector().array()/nor1

    # coulomb potential 

    b1s=frx1s*frx1s*v*dx
    a=0.25/pi*inner(nabla_grad(u), nabla_grad(v))*dx

    problem = LinearVariationalProblem(a, b1s, v1s, bc)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()

   
    # iterate Helmholtz equation

    v_xc=Function(V)
    v_xc.vector()[:] = -((6.0/pi*(frx1s.vector().array()**2.0))**(1.0/3.0))

    
    hela1s=0.5*inner(nabla_grad(u), nabla_grad(v))*dx-u*r1s*v*dx
    helb1s=(-v1s*frx1s+v_ext_corr*frx1s-v_xc*frx1s)*v*dx

    problem = LinearVariationalProblem(hela1s, helb1s, tmp1, bc)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()
   
    t1=assemble(inner(-v1s*frx1s+v_ext_corr*frx1s-v_xc*frx1s,frx1s-tmp1)*dx)
    n1=assemble(tmp1*tmp1*dx)
    r1s_p=r1s
    r1s=r1s+t1/n1
    
    time1=datetime.datetime.now()

    energy_T=0.5*assemble(inner(grad(frx1s),grad(frx1s))*dx)
    energy_H=assemble(v1s*frx1s*frx1s*dx)
    energy_ex=-assemble(v_ext_corr*frx1s*frx1s*dx)
    energy_xc=3.0/4.0*assemble(v_xc*frx1s*frx1s*dx)
    energy1=energy_T+energy_H+energy_ex+energy_xc
    energy2=r1s-0.5*energy_H-1.0/4.0*energy_xc

    # output in each step

    print r1s_p,r1s,r1s-r1s_p,energy1,energy2,(time1-time0).total_seconds()
    

