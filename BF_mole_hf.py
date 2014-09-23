
# ************************************************************************
# File:    BF_mole_hf.py
# Purpose: Solve the ground-state of BF molecule in Hartree-Fock model, including exact Fock matrix diagonalization step
# Version: FEniCS 1.1.0
# Author:  Houdong Hu
# ************************************************************************

from dolfin import *
from math import *
from numpy import *
import datetime
import time
import gc
import copy

def findex(i,j):
    k=0
    if i<j:
        t=j
        j=i
        i=t
    for t in range(i):
        k+=(t+1)
    k+=j
    return k

noe=7
nob=10

bl=2.386/2.0

mesh = Box(-50,-50,-50,50,50,50,2,2,2)

# Construct priori adaptive mesh

origin1 = Point(-bl,.0,.0)
origin2 = Point(bl,.0,.0)

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


for k in range(8):
    cell_markers = CellFunction("bool",mesh)
    cell_markers.set_all(True)
    mesh = refine(mesh, cell_markers)

File("mesh.50.50.8.xml.gz")<<mesh

cmin = 100
for cell in cells(mesh):
    p = cell.midpoint()
    r =((p[0]-origin2[0])**2+(p[1]-origin2[1])**2+(p[2]-origin2[2])**2)**0.5
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

v_ext=Expression("9.0/(pow(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2),0.5))+5.0/(pow(pow(x[0]+bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2),0.5))",bl=bl)
v_ext1 = Expression("1.0/pow(pow(x[0]-0.0,2)+pow(x[1]-0.0, 2)+pow(x[2]-0.0,2), 0.5)")
bc1 = DirichletBC(V, v_ext1, boundary)

# Define test function and initial condition (STO-3G)
u = TrialFunction(V)
v = TestFunction(V)

alpha1s=[[48.791113,8.8873622,2.405267],[166.67913,30.360812,8.2168207]]
coef1s=[0.15432897,0.53532814,0.44463454]

alpha2s=[[2.2369561,0.5198205,0.1690618],[6.4648032,1.5022812,0.4885885]]
coef2s=[-0.09996723,0.39951283,0.70011547]

alpha2p=[[2.2369561,0.5198205,0.1690618],[6.4648032,1.5022812,0.4885885]]
coef2p=[0.15591627,0.60768372,0.39195739]

tmp=[-bl,bl]
tmpv=interpolate(u0,V)
i1s=[tmpv,tmpv]
i2s=[tmpv,tmpv]
i2px=[tmpv,tmpv]
i2py=[tmpv,tmpv]
i2pz=[tmpv,tmpv]

for i in range(2):

    tmpf1=Expression("(pow(2*alpha1/pi,3.0/4.0))*exp(-alpha1*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha1=alpha1s[i][0],bl=tmp[i])
    tmpf2=Expression("(pow(2*alpha2/pi,3.0/4.0))*exp(-alpha2*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha2=alpha1s[i][1],bl=tmp[i])
    tmpf3=Expression("(pow(2*alpha3/pi,3.0/4.0))*exp(-alpha3*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha3=alpha1s[i][2],bl=tmp[i])
    i1s[i]=coef1s[0]*(interpolate(tmpf1,V)).vector().array()+coef1s[1]*(interpolate(tmpf2,V)).vector().array()+coef1s[2]*(interpolate(tmpf3,V)).vector().array()

    tmpf1=Expression("(pow(2*alpha1/pi,3.0/4.0))*exp(-alpha1*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha1=alpha2s[i][0],bl=tmp[i])
    tmpf2=Expression("(pow(2*alpha2/pi,3.0/4.0))*exp(-alpha2*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha2=alpha2s[i][1],bl=tmp[i])
    tmpf3=Expression("(pow(2*alpha3/pi,3.0/4.0))*exp(-alpha3*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha3=alpha2s[i][2],bl=tmp[i])
    i2s[i]=coef2s[0]*(interpolate(tmpf1,V)).vector().array()+coef2s[1]*(interpolate(tmpf2,V)).vector().array()+coef2s[2]*(interpolate(tmpf3,V)).vector().array()

    tmpf1=Expression("(pow(2*alpha1/pi,3.0/4.0))*exp(-alpha1*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha1=alpha2p[i][0],bl=tmp[i])
    tmpf2=Expression("(pow(2*alpha2/pi,3.0/4.0))*exp(-alpha2*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha2=alpha2p[i][1],bl=tmp[i])
    tmpf3=Expression("(pow(2*alpha3/pi,3.0/4.0))*exp(-alpha3*(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2)))",alpha3=alpha2p[i][2],bl=tmp[i])
    tmpcos1=Expression("(x[0]-bl)/pow(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2),0.5)",bl=tmp[i])
    tmpsin1=Expression("pow(pow(x[1]-0.0,2)+pow(x[2]-0.0,2),0.5)/pow(pow(x[0]-bl,2)+pow(x[1]-0.0,2)+pow(x[2]-0.0,2),0.5)",bl=tmp[i])
    tmpcos2=Expression("x[1]/(pow(pow(x[1]-0.0,2)+pow(x[2]-0.0,2),0.5)+delta)",delta=cmin)
    tmpsin2=Expression("x[2]/(pow(pow(x[1]-0.0,2)+pow(x[2]-0.0,2),0.5)+delta)",delta=cmin)

    i2px[i]=(coef2p[0]*(interpolate(tmpf1,V)).vector().array()+coef2p[1]*(interpolate(tmpf2,V)).vector().array()+coef2p[2]*(interpolate(tmpf3,V)).vector().array())*((interpolate(tmpsin1,V)).vector().array())*((interpolate(tmpsin2,V)).vector().array())
    i2py[i]=(coef2p[0]*(interpolate(tmpf1,V)).vector().array()+coef2p[1]*(interpolate(tmpf2,V)).vector().array()+coef2p[2]*(interpolate(tmpf3,V)).vector().array())*((interpolate(tmpsin1,V)).vector().array())*((interpolate(tmpcos2,V)).vector().array())
    i2pz[i]=(coef2p[0]*(interpolate(tmpf1,V)).vector().array()+coef2p[1]*(interpolate(tmpf2,V)).vector().array()+coef2p[2]*(interpolate(tmpf3,V)).vector().array())*(interpolate(tmpcos1,V)).vector().array()

for tmpv in [i1s,i2s,i2px,i2py,i2pz]:
    for i in range(2):
        tmp=Function(V)
        tmp.vector()[:]=tmpv[i]
        nor=assemble(inner(tmp,tmp)*dx)
        tmpv[i]=tmpv[i]/(nor**0.5)

im1=0.992073*i1s[1]
im2=0.992073*i1s[0]
im3=-0.250307*i1s[1]+0.940298*i2s[1]
im4=0.384238*i2s[1]+0.803403*i2pz[1]-0.36005*i2s[0]-0.180433*i2pz[0]
im5=0.871574*i2px[1]-0.317707*i2py[1]+0.230384*i2px[0]
im6=0.317707*i2px[1]+0.871574*i2py[1]+0.230384*i2py[0]
im7=-0.24306*i1s[0]+0.902268*i2s[0]-0.462384*i2pz[0]+0.282457*i2pz[1]



rx=[im1,im2,im3,im4,im5,im6,im7]
r=[-26.0364,-7.3490,-1.5806,-0.6769,-0.573,-0.573,-0.2797]
r_p=zeros(noe)
energy_p=0.0
energy=-2.5

vpof=[]
for i in range(noe):
    for j in range(i+1):
        tmp=Function(V)
        tmp.vector()[:]=interpolate(u0,V).vector().array()
        vpof.append(tmp)

frx=[]
for i in range(noe):
    tmp=Function(V)
    tmp.vector()[:]=rx[i]
    nor=assemble(inner(tmp,tmp)*dx)
    tmp.vector()[:]=rx[i]/(nor**0.5)
    frx.append(tmp)

threshold=1e-7
threshold2=0.1

while abs(energy-energy_p)>threshold:

    time0=datetime.datetime.now()
    
    tmp1=Function(V)
    tmp2=Function(V)

    # potential solver

    for i in range(noe):
        for j in range(i+1):
            tmp=Function(V)
            tmpb=frx[i]*frx[j]*v*dx
            a=0.25/pi*inner(nabla_grad(u), nabla_grad(v))*dx
            if i==j:
                problem = LinearVariationalProblem(a, tmpb, tmp, bc1)
            else:
                problem = LinearVariationalProblem(a, tmpb, tmp, bc)
            solver = LinearVariationalSolver(problem)
            solver.parameters["linear_solver"] = "cg"
            solver.parameters["preconditioner"] = "amg"
            solver.solve()
            vpof[findex(i,j)].vector()[:]=tmp.vector().array()

    # Fock matrix setup and diagonalizatin
   
    mat=zeros((noe,noe))

    for i in range(noe):
        for j in range(i+1):
            mat[i][j]=0.5*assemble(inner(grad(frx[i]),grad(frx[j]))*dx)-assemble(v_ext*frx[i]*frx[j]*dx)
            for k in range(noe):
                mat[i][j]+=2.0*assemble(vpof[findex(k,k)]*frx[i]*frx[j]*dx)
                mat[i][j]-=assemble(vpof[findex(k,j)]*frx[i]*frx[k]*dx)
            mat[j][i]=mat[i][j]

    
    r_p=copy.deepcopy(r)
    rx_p=copy.deepcopy(rx)
    ev,ef=linalg.eig(mat)
    print mat,r,ev,ef

    for i in range(noe):
        cmax=0.0 
        index=0
        rx[i]=interpolate(u0,V).vector().array()
        for j in range(noe):
            rx[i]+=ef[j,i]*rx_p[j]
            if ef[j,i]**2.0>cmax:
                index=j
                cmax=ef[j,i]**2.0
        frx[i].vector()[:]=rx[i]
        r[i]=r_p[index]
        print i,index
 
    if abs(energy-energy_p)<threshold2:
        r=ev
    
    # compute potential

    for i in range(noe):
        for j in range(i+1):
            tmpb=frx[i]*frx[j]*v*dx
            a=0.25/pi*inner(nabla_grad(u), nabla_grad(v))*dx
            if i==j:
                problem = LinearVariationalProblem(a, tmpb, tmp, bc1)
            else:
                problem = LinearVariationalProblem(a, tmpb, tmp, bc)
            solver = LinearVariationalSolver(problem)
            solver.parameters["linear_solver"] = "gmres"
            solver.parameters["preconditioner"] = "amg"
            solver.solve()
            vpof[findex(i,j)].vector()[:]=tmp.vector().array()

    # iterate Helmholtz equation

    rx_p=copy.deepcopy(rx)
   
    for i in range(noe):
        
        tmp=Function(V)
        hela=0.5*inner(nabla_grad(u), nabla_grad(v))*dx-u*r[i]*v*dx
        helb=v_ext*frx[i]*v*dx
        for j in range(noe):
            helb+=(-2.0*vpof[findex(j,j)]*frx[i]*v*dx+vpof[findex(j,i)]*frx[j]*v*dx)

        problem = LinearVariationalProblem(hela, helb, tmp, bc)
        solver = LinearVariationalSolver(problem)
        solver.parameters["linear_solver"] = "gmres"
        solver.parameters["preconditioner"] = "amg"
        solver.solve()

        rx[i]=tmp.vector().array()
        print i,assemble(inner(tmp-frx[i],tmp-frx[i])*dx)


    for i in range(noe):
        for j in range(i):
            tmp1.vector()[:]=rx[i]     
            tmp2.vector()[:]=rx[j]     
            nor=assemble(inner(tmp1,tmp2)*dx)
            print i,j,nor
            rx[i]-=nor*rx[j]
            
        tmp1.vector()[:]=rx[i]
        nor=assemble(inner(tmp1,tmp1)*dx)
        rx[i]=rx[i]/(nor**0.5)
        frx[i].vector()[:]=rx[i]
 
    energy_p=energy
    energy_T=0.0
    energy_ext=0.0
    energy_H=0.0
    energy_ec=0.0

    for i in range(noe):
        tmp1.vector()[:]=rx_p[i]
        energy_T+=assemble(inner(grad(tmp1),grad(tmp1))*dx)
        energy_ext-=2.0*assemble(v_ext*tmp1*tmp1*dx)
        for j in range(noe):
            tmp2.vector()[:]=rx_p[j]
            energy_H+=2.0*assemble(vpof[findex(j,j)]*tmp1*tmp1*dx)
            energy_ec-=assemble(vpof[findex(j,i)]*tmp1*tmp2*dx)
    

    energy= energy_T+energy_ec+energy_ext+energy_H
    energy+=45.0/(2.0*bl)
    print r    

    time1=datetime.datetime.now() 
    
    # output in each step
    
    print energy_p,energy,energy-energy_p,(time1-time0).total_seconds()

