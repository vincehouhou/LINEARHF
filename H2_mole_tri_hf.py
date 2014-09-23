
# ************************************************************************
# File:    H2_mole_tri_hf.py
# Purpose: Solve the ground-state of H2 molecule triplet state in Hartree-Fock model, including exact Fock matrix diagonalization step
# Version: FEniCS 1.1.0
# Author:  Houdong Hu
# ************************************************************************


from dolfin import *
from math import *
from numpy import *
import datetime

bl=1.4

mesh = Box(-50,-50,-50,50,50,50,2,2,2)
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
        r=((p[0]-origin1[0])**2+(p[1]-origin1[1])**2+(p[2]-origin1[2])**2)**0.5
        if r<min1:
            min1=r
        r=((p[0]-origin2[0])**2+(p[1]-origin2[1])**2+(p[2]-origin2[2])**2)**0.5
        if r<min2:
            min2=r

    for cell in cells(mesh):
        p = cell.midpoint()
        r=((p[0]-origin1[0])**2+(p[1]-origin1[1])**2+(p[2]-origin1[2])**2)**0.5
        if r<=min1:
            cell_markers[cell]=True
        r=((p[0]-origin2[0])**2+(p[1]-origin2[1])**2+(p[2]-origin2[2])**2)**0.5
        if r<=min2:
            cell_markers[cell]=True

    mesh=refine(mesh,cell_markers)


for k in range(4):
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


V = FunctionSpace(mesh, "Lagrange", 1)

def boundary(x, on_boundary):
    return on_boundary

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

v_ext=Expression("1.0/(pow(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2),0.5))+1.0/(pow(pow(x[0]+bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2),0.5))",bl=bl)
v_ext1 = Expression("1.0/pow(pow(x[0]-0.0,2)+pow(x[1]-0.0, 2)+pow(x[2]-0.0,2), 0.5)")
bc1 = DirichletBC(V, v_ext1, boundary)

# Define test function and initial condition (STO-3G)
u = TrialFunction(V)
v = TestFunction(V)

alpha11=3.43525091
alpha12=0.62391373
alpha13=0.16885540
tmp11=Expression("(pow(2*alpha11/pi,3.0/4.0))*exp(-alpha11*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha11=alpha11,bl=bl)
tmp12=Expression("(pow(2*alpha12/pi,3.0/4.0))*exp(-alpha12*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha12=alpha12,bl=bl)
tmp13=Expression("(pow(2*alpha13/pi,3.0/4.0))*exp(-alpha13*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha13=alpha13,bl=bl)
rx11s=0.15432897*(interpolate(tmp11,V)).vector().array()+0.53532814*(interpolate(tmp12,V)).vector().array()+0.44463454*(interpolate(tmp13,V)).vector().array()


tmp11=Expression("(pow(2*alpha11/pi,3.0/4.0))*exp(-alpha11*(pow(x[0]+bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha11=alpha11,bl=bl)
tmp12=Expression("(pow(2*alpha12/pi,3.0/4.0))*exp(-alpha12*(pow(x[0]+bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha12=alpha12,bl=bl)
tmp13=Expression("(pow(2*alpha13/pi,3.0/4.0))*exp(-alpha13*(pow(x[0]+bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha13=alpha13,bl=bl)
rx22s=0.15432897*(interpolate(tmp11,V)).vector().array()+0.53532814*(interpolate(tmp12,V)).vector().array()+0.44463454*(interpolate(tmp13,V)).vector().array()

r1s=-0.5
r1s_p=0.0
r2s=-0.5
r2s_p=0.0
energy=-2.5
rx1s=0.5*rx11s+0.5*rx22s
rx2s=0.5*rx11s-0.5*rx22s

tmp1=Function(V)
tmp1.vector()[:]=rx1s
tmp2=Function(V)
tmp2.vector()[:]=rx2s

v1s = Function(V)
v2s = Function(V)
v12 = Function(V)
frx1s=Function(V)
frx2s=Function(V)

while True:

    time0=datetime.datetime.now()

    # orthogonization
    nor1=assemble(inner(tmp1,tmp1)*dx)
    nor12=assemble(inner(tmp1,tmp2)*dx)
    rx2s=rx2s-nor12/nor1*rx1s
    tmp2.vector()[:]=rx2s

    # normalization
    nor1=assemble(inner(tmp1,tmp1)*dx)**0.5
    nor2=assemble(inner(tmp2,tmp2)*dx)**0.5
    rx1s=rx1s/nor1
    rx2s=rx2s/nor2

    frx1s.vector()[:]=rx1s
    frx2s.vector()[:]=rx2s

    
    # potential solver
    b1s=frx1s*frx1s*v*dx
    b2s=frx2s*frx2s*v*dx
    b12=frx1s*frx2s*v*dx
    a=0.25/pi*inner(nabla_grad(u), nabla_grad(v))*dx

    v1s = Function(V)
    problem = LinearVariationalProblem(a, b1s, v1s, bc1)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()

    v2s = Function(V)
    problem = LinearVariationalProblem(a, b2s, v2s, bc1)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()

    v12 = Function(V)
    problem = LinearVariationalProblem(a, b12, v12, bc)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()

    # Fock matrix setup and diagonalization
    mat11=0.5*assemble(inner(grad(frx1s),grad(frx1s))*dx)+assemble((v1s+v2s)*frx1s*frx1s*dx)-assemble(v12*frx1s*frx2s*dx)-assemble(v1s*frx1s*frx1s*dx)-assemble(v_ext*frx1s*frx1s*dx)
    mat12=0.5*assemble(inner(grad(frx1s),grad(frx2s))*dx)+assemble((v1s+v2s)*frx1s*frx2s*dx)-assemble(v2s*frx1s*frx2s*dx)-assemble(v12*frx1s*frx1s*dx)-assemble(v_ext*frx1s*frx2s*dx)
    mat21=0.5*assemble(inner(grad(frx1s),grad(frx2s))*dx)+assemble((v1s+v2s)*frx1s*frx2s*dx)-assemble(v1s*frx2s*frx1s*dx)-assemble(v12*frx2s*frx2s*dx)-assemble(v_ext*frx1s*frx2s*dx)
    mat22=0.5*assemble(inner(grad(frx2s),grad(frx2s))*dx)+assemble((v1s+v2s)*frx2s*frx2s*dx)-assemble(v12*frx1s*frx2s*dx)-assemble(v2s*frx2s*frx2s*dx)-assemble(v_ext*frx2s*frx2s*dx)

    r1s_p=r1s
    r2s_p=r2s

    mat=array([[mat11,mat12],[mat21,mat22]])
    ev,ef=linalg.eig(mat)
    ind1=0
    ind2=1
    if ev[0]>ev[1]:
        ind1=1
        ind2=0

    r1s=ev[ind1]
    r2s=ev[ind2]
    tmp3=ef[0,ind1]*rx1s+ef[1,ind1]*rx2s
    rx2s=ef[0,ind2]*rx1s+ef[1,ind2]*rx2s
    rx1s=tmp3

    frx1s.vector()[:]=rx1s
    frx2s.vector()[:]=rx2s

    print r1s,r2s

    # compute potential

    b1s=frx1s*frx1s*v*dx
    b2s=frx2s*frx2s*v*dx
    b12=frx1s*frx2s*v*dx
    a=0.25/pi*inner(nabla_grad(u), nabla_grad(v))*dx

    problem = LinearVariationalProblem(a, b1s, v1s, bc1)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()

    problem = LinearVariationalProblem(a, b2s, v2s, bc1)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()

    problem = LinearVariationalProblem(a, b12, v12, bc)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "amg"
    solver.solve()

    # iterate Helmholtz equation

    hela1s=0.5*inner(nabla_grad(u), nabla_grad(v))*dx-u*r1s*v*dx
    hela2s=0.5*inner(nabla_grad(u), nabla_grad(v))*dx-u*r2s*v*dx
    helb1s=(-(v1s+v2s)*frx1s+v12*frx2s+v1s*frx1s+v_ext*frx1s)*v*dx
    helb2s=(-(v1s+v2s)*frx2s+v12*frx1s+v2s*frx2s+v_ext*frx2s)*v*dx

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

    rx1s=tmp1.vector().array()
    rx2s=tmp2.vector().array()

    energy_T=0.5*assemble(inner(grad(frx1s),grad(frx1s))*dx)+0.5*assemble(inner(grad(frx2s),grad(frx2s))*dx)    
    energy_H=assemble((v1s+v2s)*frx1s*frx1s*dx)+assemble((v1s+v2s)*frx2s*frx2s*dx)
    energy_ext=-assemble(v_ext*frx1s*frx1s*dx)-assemble(v_ext*frx2s*frx2s*dx)
    energy_xc=-assemble(2.0*frx1s*frx2s*v12*dx)-assemble(frx1s*frx1s*v1s*dx)-assemble(frx2s*frx2s*v2s*dx)
    energy_nuc=1.0/(2.0*bl)
    energy_p=energy
    energy=energy_T+energy_H+energy_ext+energy_xc+energy_nuc
    

    time1=datetime.datetime.now() 
    
    # output in each step

    print r1s,r2s,energy_p,energy,energy-energy_p,(time1-time0).total_seconds(),assemble(frx1s*frx1s*dx),assemble(frx2s*frx2s*dx),assemble(frx1s*frx2s*dx),assemble(frx1s*dx),assemble(frx2s*dx)
