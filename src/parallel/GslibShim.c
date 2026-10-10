/* SPDX-License-Identifier: LGPL-2.1-or-later */
/*
 * The gather-scatter of gslib, the library of Nek5000, behind four C functions.
 * gslib's headers are C99: they define inline and restrict away in other dialects, so only this C file includes them and the C++ code calls the functions below.
 */
#include <mpi.h>
#include <stdlib.h>

#include "gslib.h"

struct dualmesh_gslib
{
  struct comm comm;
  struct gs_data * handle;
};

/* Set up the gather-scatter of the local entities whose global indices are ids[0..n), all positive. */
void *
dualmesh_gslib_setup(MPI_Comm communicator, const long long * ids, unsigned n)
{
  struct dualmesh_gslib * self = malloc(sizeof(struct dualmesh_gslib));
  comm_init(&self->comm, communicator);
  /* gs_auto times the pairwise, crystal router and all-reduce exchanges and keeps the fastest. */
  self->handle = gs_setup((const slong *)ids, n, &self->comm, 0, gs_auto, 0);
  return self;
}

/* Replace each of the width values of every entity by its sum over the ranks that hold the entity. */
void
dualmesh_gslib_sum(void * self, double * values, unsigned width)
{
  struct dualmesh_gslib * g = self;
  if (width == 1)
    gs(values, gs_double, gs_add, 0, g->handle, NULL);
  else
    gs_vec(values, width, gs_double, gs_add, 0, g->handle, NULL);
}

/* Replace the value of every entity by its minimum over the ranks that hold the entity. */
void
dualmesh_gslib_min(void * self, int * values)
{
  struct dualmesh_gslib * g = self;
  gs(values, gs_int, gs_min, 0, g->handle, NULL);
}

void
dualmesh_gslib_free(void * self)
{
  struct dualmesh_gslib * g = self;
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized)
  {
    gs_free(g->handle);
    comm_free(&g->comm);
  }
  free(g);
}
