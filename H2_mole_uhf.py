
# ************************************************************************
# File:    H2_mole_uhf.py
# Purpose: Solve the ground-state of H2 molecue in unrestricted Hartree-Fock model, including energy correction step
# Version: FEniCS 1.4.0
# Author:  Houdong Hu
# ************************************************************************


from dolfin import *
from math import *
import numpy as np
import datetime

bl=1.2

mesh = BoxMesh(-50,-50,-50,50,50,50,2,2,2)
origin1 = Point(-bl,.0,.0)
origin2 = Point(bl,.0,.0)

# construct priori adaptive mesh

for j in range(50):
    cell_markers = CellFunction("bool",mesh)
    cell_markers.set_all(False)
    min1 = 1000
    min2 = 1000
    for cell in cells(mesh):
        p = cell.midpoint()
        r =((p[0]-origin1[0])**2+(p[1]-origin1[1])**2+(p[2]-origin1[2])**2)**0.5
        if r<min1:
           min1=r
        r =((p[0]-origin2[0])**2+(p[1]-origin2[1])**2+(p[2]-origin2[2])**2)**0.5
        if r<min2:
           min2=r
       
    for cell in cells(mesh):
        p = cell.midpoint()
        r =((p[0]-origin1[0])**2+(p[1]-origin1[1])**2+(p[2]-origin1[2])**2)**0.5
        if r<=min1:
            cell_markers[cell]=True
        r =((p[0]-origin2[0])**2+(p[1]-origin2[1])**2+(p[2]-origin2[2])**2)**0.5
        if r<=min2:
            cell_markers[cell]=True
   
   
    mesh = refine(mesh, cell_markers)

for k in range(6):
    cell_markers = CellFunction("bool",mesh)
    cell_markers.set_all(True)
    mesh = refine(mesh, cell_markers)

cmin = 100
for cell in cells(mesh):
    p = cell.midpoint()
    r =((p[0]-origin1[0])**2+(p[1]-origin1[1])**2+(p[2]-origin1[2])**2)**0.5
    if r<cmin:
       cmin=r
print cmin
print mesh.coordinates().shape

V = FunctionSpace(mesh, "Lagrange", 1)

delta=cmin
v_ext_corr=Expression("1.0/(pow(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2),0.5))+1.0/(pow(pow(x[0]+bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2),0.5))",bl=bl)#+delta)",delta=delta)
v_ext1=Expression("1.0/pow(pow(x[0]-0.0,2)+pow(x[1]-0.0, 2)+pow(x[2]-0.0,2),0.5)")

def boundary(x, on_boundary):
    return on_boundary

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)
bc1 = DirichletBC(V, v_ext1, boundary)


# Define test function and initial condition (STO-3G)
u = TrialFunction(V)
v = TestFunction(V)

alpha11=3.43525091
alpha12=0.62391373
alpha13=0.16885540
tmp11=Expression("(pow(2*alpha11/pi,3.0/4.0))*exp(-alpha11*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",bl=bl,alpha11=alpha11)
tmp12=Expression("(pow(2*alpha12/pi,3.0/4.0))*exp(-alpha12*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",bl=bl,alpha12=alpha12)
tmp13=Expression("(pow(2*alpha13/pi,3.0/4.0))*exp(-alpha13*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",bl=bl,alpha13=alpha13)
rx1s=0.15432897*(interpolate(tmp11,V)).vector().array()+0.53532814*(interpolate(tmp12,V)).vector().array()+0.44463454*(interpolate(tmp13,V)).vector().array()

tmp11=Expression("(pow(2*alpha11/pi,3.0/4.0))*exp(-alpha11*(pow(x[0]+bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",bl=bl,alpha11=alpha11)
tmp12=Expression("(pow(2*alpha12/pi,3.0/4.0))*exp(-alpha12*(pow(x[0]+bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",bl=bl,alpha12=alpha12)
tmp13=Expression("(pow(2*alpha13/pi,3.0/4.0))*exp(-alpha13*(pow(x[0]+bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",bl=bl,alpha13=alpha13)
rx2s=0.15432897*(interpolate(tmp11,V)).vector().array()+0.53532814*(interpolate(tmp12,V)).vector().array()+0.44463454*(interpolate(tmp13,V)).vector().array()


r1s=-0.5
r1s_p=0.0
r2s=-0.5
r2s_p=0.0
energy=-2.5

tmp1=Function(V)
tmp1.vector()[:]=rx1s
frx1s=Function(V)
v1s = Function(V)
tmp2=Function(V)
tmp2.vector()[:]=rx2s
frx2s=Function(V)
v2s = Function(V)
    
while True:

    time0=datetime.datetime.now()

    # normalization
    nor1=assemble(inner(tmp1,tmp1)*dx)**0.5
    frx1s.vector()[:]=tmp1.vector().array()/nor1
    nor2=assemble(inner(tmp2,tmp2)*dx)**0.5
    frx2s.vector()[:]=tmp2.vector().array()/nor2

    # coulomb potential 

    b1s=frx1s*frx1s*v*dx
    a=0.25/pi*inner(nabla_grad(u), nabla_grad(v))*dx

    problem = LinearVariationalProblem(a, b1s, v1s, bc1)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()
    
    b2s=frx2s*frx2s*v*dx

    problem = LinearVariationalProblem(a, b2s, v2s, bc1)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()
 
    # iterate Helmholtz equation

    hela1s=0.5*inner(nabla_grad(u), nabla_grad(v))*dx-u*r1s*v*dx
    helb1s=(-v2s*frx1s+v_ext_corr*frx1s)*v*dx
    hela2s=0.5*inner(nabla_grad(u), nabla_grad(v))*dx-u*r2s*v*dx
    helb2s=(-v1s*frx2s+v_ext_corr*frx2s)*v*dx

    problem = LinearVariationalProblem(hela1s, helb1s, tmp1, bc)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()
    problem = LinearVariationalProblem(hela2s, helb2s, tmp2, bc)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()
   
    # energy correction step

    t1=assemble(inner(-v2s*frx1s+v_ext_corr*frx1s,frx1s-tmp1)*dx)
    n1=assemble(tmp1*tmp1*dx)
    r1s_p=r1s
    r1s=r1s+t1/n1

    t2=assemble(inner(-v1s*frx2s+v_ext_corr*frx2s,frx2s-tmp2)*dx)
    n2=assemble(tmp2*tmp2*dx)
    r2s_p=r2s
    r2s=r2s+t2/n2
 
    energy_T=0.5*assemble(inner(grad(frx1s),grad(frx1s))*dx)+0.5*assemble(inner(grad(frx2s),grad(frx2s))*dx)
    energy_H=0.5*assemble(v2s*frx1s*frx1s*dx)+0.5*assemble(v1s*frx2s*frx2s*dx)
    energy_ex=-assemble(v_ext_corr*frx1s*frx1s*dx)-assemble(v_ext_corr*frx2s*frx2s*dx)
    energy_nuc=1.0/(2.0*bl)
    energy_p=energy
    energy=energy_T+energy_H+energy_ex+energy_nuc
    
    time1=datetime.datetime.now()

    # output in each step

    print r1s,r2s,energy_p,energy,energy-energy_p,(time1-time0).total_seconds(),assemble(frx1s*frx1s*dx),assemble(frx2s*frx2s*dx),assemble(frx1s*frx2s*dx)
  
    

