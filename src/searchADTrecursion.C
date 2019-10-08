//
// This file is part of the Tioga software library
//
// Tioga  is a tool for overset grid assembly on parallel distributed systems
// Copyright (C) 2015 Jay Sitaraman
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#include "codetypes.h"
#include "MeshBlock.h"

void searchIntersections(MeshBlock *mb,int *cellIndex,int *adtIntegers,double *adtReals,
			 double *coord,int level,int node,double *xsearch,int nelem,int ndim);

void ADT::searchADT(MeshBlock *mb, int *cellIndex,double *xsearch)
{
#ifdef TIOGA_USE_ARBORX

  using QueryType = ArborX::Intersects<ArborX::Point>;

  // Setup queries
  const int n_queries = 1;
  Kokkos::View<QueryType *, DeviceType> queries(
      Kokkos::ViewAllocateWithoutInitializing("queries"), n_queries);

  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::parallel_for("bvh_driver:setup_radius_search_queries",
                       Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
                       KOKKOS_LAMBDA(int i) {
                         queries(i) = QueryType(ArborX::Point{xsearch[0],xsearch[1],xsearch[2]});
                       });

  Kokkos::View<int *, DeviceType> offset("offset", 0);
  Kokkos::View<int *, DeviceType> indices("indices", 0);
  bvh.query(queries, indices, offset, 0/*buffer_size*/);

  int ncandidates = offset.size() - 1;     // number of approximate neighbors
  int *candidateList = indices.data();    // index of candidate AABB

  for(int j = 0; j < ncandidates; j++)
    {
      mb->checkContainment(cellIndex, candidateList[j], xsearch);
      // checkContainment will map the element in adt list to the actual cell
      if (cellIndex[0] > -1 && cellIndex[1]==0) return;
    }
#else
  int i;
  int flag;
  int rootNode;
  //
  // check if the given point is in the bounds of
  // the ADT
  //
  rootNode=0;
  cellIndex[0]=-1;
  cellIndex[1]=0;
  //
  flag=1;
  for(i=0;i<ndim/2;i++)
    flag = (flag && (xsearch[i] >= adtExtents[2*i]-TOL));
  for(i=0;i<ndim/2;i++)
    flag= (flag && (xsearch[i] <= adtExtents[2*i+1]+TOL));
  //
  // call recursive routine to check intersections with
  // ADT nodes
  //
  if (flag) searchIntersections(mb,cellIndex,adtIntegers,adtReals,
				coord,0,rootNode,xsearch,nelem,ndim);
#endif
}

void searchIntersections(MeshBlock *mb,int *cellIndex,int *adtIntegers,double *adtReals,
			 double *coord,int level,int node,double *xsearch,int nelem,int ndim)
{
  int i;
  int d,nodeChild,dimcut;
  double element[ndim];
  bool flag;
  //
  for(i=0;i<ndim;i++)
    element[i]=coord[ndim*(adtIntegers[4*node])+i];
  //
  flag=1;
  for(i=0;i<ndim/2;i++)
    flag = (flag && (xsearch[i] >=element[i]-TOL));
  for(i=ndim/2;i<ndim;i++)
    flag = (flag && (xsearch[i-ndim/2] <=element[i]+TOL));
  //
  if (flag)
    {
      mb->checkContainment(cellIndex,adtIntegers[4*node],xsearch);
      if (cellIndex[0] > -1 && cellIndex[1]==0) return;
    }
  //
  // check the left and right children
  // now
  //
  for(d=1;d<3;d++)
    {
      nodeChild=adtIntegers[4*node+d];
      if (nodeChild > -1) {
        nodeChild=adtIntegers[4*nodeChild+3];
	for(i=0;i<ndim;i++)
         {
	  element[i]=adtReals[ndim*nodeChild+i];
         }
	flag=1;
	for(i=0;i<ndim/2;i++)
	  flag = (flag && (xsearch[i] >=element[i]-TOL));
	for(i=ndim/2;i<ndim;i++)
	  flag = (flag && (xsearch[i-ndim/2] <=element[i]+TOL));
	if (flag)
	  {
	    searchIntersections(mb,cellIndex,adtIntegers,adtReals,coord,level+1,
			       nodeChild,xsearch,nelem,ndim);
	    if (cellIndex[0] > -1 && cellIndex[1]==0) return;
	  }
      }
    }
  return;
}
