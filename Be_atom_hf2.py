
# ************************************************************************
# File:    Be_atom_hf2.py
# Purpose: Solve the ground-state of beryllium atom in Hartree-Fock model, including energy correction step
# Version: FEniCS 1.1.0
# Author:  Houdong Hu
# ************************************************************************


from dolfin import *
from math import *
from numpy import *
import datetime

mesh = Box(-40,-40,-40,40,40,40,2,2,2)
origin = Point(.0,.0,.0)

# Construct priori adaptive mesh

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

for k in range(2):
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


V = FunctionSpace(mesh, "Lagrange", 1)

def boundary(x, on_boundary):
    return on_boundary

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

v_ext = Expression("4.0/pow(pow(x[0]-0.0,2)+pow(x[1]-0.0, 2)+pow(x[2]-0.0,2), 0.5)")
v_ext1 = Expression("1.0/pow(pow(x[0]-0.0,2)+pow(x[1]-0.0, 2)+pow(x[2]-0.0,2), 0.5)")
bc1 = DirichletBC(V, v_ext1, boundary)

delta=cmin
v_ext_corr = Expression("4.0/(pow(pow(x[0]-0.0,2)+pow(x[1]-0.0, 2)+pow(x[2]-0.0,2), 0.5)+delta)",delta=delta)


# Define test function and initial condition (STO-3G)
u = TrialFunction(V)
v = TestFunction(V)

alpha11=30.1678710
alpha12=5.4951153
alpha13=1.4871927
alpha21=1.3148331
alpha22=0.3055389
alpha23=0.0993707
tmp11=Expression("(pow(2*alpha11/pi,3.0/4.0))*exp(-alpha11*(pow(x[0]-0.0,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha11=alpha11)
tmp12=Expression("(pow(2*alpha12/pi,3.0/4.0))*exp(-alpha12*(pow(x[0]-0.0,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha12=alpha12)
tmp13=Expression("(pow(2*alpha13/pi,3.0/4.0))*exp(-alpha13*(pow(x[0]-0.0,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha13=alpha13)
tmp21=Expression("(pow(2*alpha21/pi,3.0/4.0))*exp(-alpha21*(pow(x[0]-0.0,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha21=alpha21)
tmp22=Expression("(pow(2*alpha22/pi,3.0/4.0))*exp(-alpha22*(pow(x[0]-0.0,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha22=alpha22)
tmp23=Expression("(pow(2*alpha23/pi,3.0/4.0))*exp(-alpha23*(pow(x[0]-0.0,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha23=alpha23)


rx1s=0.15432897*(interpolate(tmp11,V)).vector().array()+0.53532814*(interpolate(tmp12,V)).vector().array()+0.44463454*(interpolate(tmp13,V)).vector().array()
rx2s=-0.09996723*(interpolate(tmp21,V)).vector().array()+0.39951283*(interpolate(tmp22,V)).vector().array()+0.70011547*(interpolate(tmp23,V)).vector().array()

r1s=-4.81
r2s=-0.24
r1s_p=0.0
r2s_p=0.0
energy=-14.0

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
    

    # step 1 - compute potential

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

    # step 2 - iterate Helmholtz equation

    hela1s=0.5*inner(nabla_grad(u), nabla_grad(v))*dx-u*r1s*v*dx
    hela2s=0.5*inner(nabla_grad(u), nabla_grad(v))*dx-u*r2s*v*dx
    helb1s=(-2.0*(v1s+v2s)*frx1s+v12*frx2s+v1s*frx1s+v_ext_corr*frx1s)*v*dx
    helb2s=(-2.0*(v1s+v2s)*frx2s+v12*frx1s+v2s*frx2s+v_ext_corr*frx2s)*v*dx


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


    # step 3 - energy correction

    t1=assemble(inner(-2.0*(v1s+v2s)*frx1s+v12*frx2s+v1s*frx1s+v_ext_corr*frx1s,frx1s-tmp1)*dx)
    n1=assemble(tmp1*tmp1*dx)
    t2=assemble(inner(-2.0*(v1s+v2s)*frx2s+v12*frx1s+v2s*frx2s+v_ext_corr*frx2s,frx2s-tmp2)*dx)
    n2=assemble(tmp2*tmp2*dx)

    r1s_p=r1s
    r2s_p=r2s
    r1s=r1s+t1/n1
    r2s=r2s+t2/n2
        
    rx1s=tmp1.vector().array()
    rx2s=tmp2.vector().array()

    energy_T=assemble(inner(grad(frx1s),grad(frx1s))*dx)+assemble(inner(grad(frx2s),grad(frx2s))*dx)    
    energy_H=assemble(2.0*(v1s+v2s)*frx1s*frx1s*dx)+assemble(2.0*(v1s+v2s)*frx2s*frx2s*dx)
    energy_ext=-assemble(2.0*v_ext_corr*frx1s*frx1s*dx)-assemble(2.0*v_ext_corr*frx2s*frx2s*dx)
    energy_xc=-assemble(2.0*frx1s*frx2s*v12*dx)-assemble(frx1s*frx1s*v1s*dx)-assemble(frx2s*frx2s*v2s*dx)
    energy_p=energy
    energy=energy_T+energy_H+energy_ext+energy_xc
    
    time1=datetime.datetime.now()

    # output in each step

    print r1s,r2s,energy_p,energy,energy-energy_p,(time1-time0).total_seconds()
