#ifndef _MY_covid_data_17C729F6172AF7E869D5F1A2C22AAF22
#define _MY_covid_data_17C729F6172AF7E869D5F1A2C22AAF22

#include <cvodes/cvodes.h>
#include <cvodes/cvodes_dense.h>
#include <cvodes/cvodes_sparse.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_sparse.h>
#include <cvodes/cvodes_klu.h>
#include <udata.h>
#include <math.h>
#include <mex.h>
#include <arInputFunctionsC.h>



 void fy_covid_data_17C729F6172AF7E869D5F1A2C22AAF22(double t, int nt, int it, int ntlink, int itlink, int ny, int nx, int nz, int iruns, double *y, double *p, double *u, double *x, double *z);
 void fystd_covid_data_17C729F6172AF7E869D5F1A2C22AAF22(double t, int nt, int it, int ntlink, int itlink, double *ystd, double *y, double *p, double *u, double *x, double *z);
 void fsy_covid_data_17C729F6172AF7E869D5F1A2C22AAF22(double t, int nt, int it, int ntlink, int itlink, double *sy, double *p, double *u, double *x, double *z, double *su, double *sx, double *sz);
 void fsystd_covid_data_17C729F6172AF7E869D5F1A2C22AAF22(double t, int nt, int it, int ntlink, int itlink, double *systd, double *p, double *y, double *u, double *x, double *z, double *sy, double *su, double *sx, double *sz);

 void fy_scale_covid_data_17C729F6172AF7E869D5F1A2C22AAF22(double t, int nt, int it, int ntlink, int itlink, int ny, int nx, int nz, int iruns, double *y_scale, double *p, double *u, double *x, double *z, double *dfzdx);
#endif /* _MY_covid_data_17C729F6172AF7E869D5F1A2C22AAF22 */



