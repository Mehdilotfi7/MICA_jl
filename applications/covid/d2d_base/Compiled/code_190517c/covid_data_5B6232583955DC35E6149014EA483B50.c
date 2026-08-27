#include "covid_data_5B6232583955DC35E6149014EA483B50.h"
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





 void fy_covid_data_5B6232583955DC35E6149014EA483B50(double t, int nt, int it, int ntlink, int itlink, int ny, int nx, int nz, int iruns, double *y, double *p, double *u, double *x, double *z){
  y[ny*nt*iruns+it+nt*0] = log(z[nz*ntlink*iruns+itlink+ntlink*9]+1.0)/log(1.0E+1);
  y[ny*nt*iruns+it+nt*1] = log(z[nz*ntlink*iruns+itlink+ntlink*4]+1.0)/log(1.0E+1);
  y[ny*nt*iruns+it+nt*2] = log(z[nz*ntlink*iruns+itlink+ntlink*5]+1.0)/log(1.0E+1);
  y[ny*nt*iruns+it+nt*3] = log(z[nz*ntlink*iruns+itlink+ntlink*8]+1.0)/log(1.0E+1);
  y[ny*nt*iruns+it+nt*4] = log(z[nz*ntlink*iruns+itlink+ntlink*10]+1.0)/log(1.0E+1);

  return;
}


 void fystd_covid_data_5B6232583955DC35E6149014EA483B50(double t, int nt, int it, int ntlink, int itlink, double *ystd, double *y, double *p, double *u, double *x, double *z){
  ystd[it+nt*0] = p[16];
  ystd[it+nt*1] = p[18];
  ystd[it+nt*2] = p[19];
  ystd[it+nt*3] = p[17];
  ystd[it+nt*4] = p[20];

  return;
}


 void fsy_covid_data_5B6232583955DC35E6149014EA483B50(double t, int nt, int it, int ntlink, int itlink, double *sy, double *p, double *u, double *x, double *z, double *su, double *sx, double *sz){
  sy[it+nt*0] = sz[itlink+ntlink*9]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*1] = sz[itlink+ntlink*4]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*2] = sz[itlink+ntlink*5]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*3] = sz[itlink+ntlink*8]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*4] = sz[itlink+ntlink*10]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*5] = sz[itlink+ntlink*24]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*6] = sz[itlink+ntlink*19]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*7] = sz[itlink+ntlink*20]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*8] = sz[itlink+ntlink*23]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*9] = sz[itlink+ntlink*25]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*10] = sz[itlink+ntlink*39]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*11] = sz[itlink+ntlink*34]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*12] = sz[itlink+ntlink*35]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*13] = sz[itlink+ntlink*38]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*14] = sz[itlink+ntlink*40]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*15] = sz[itlink+ntlink*54]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*16] = sz[itlink+ntlink*49]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*17] = sz[itlink+ntlink*50]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*18] = sz[itlink+ntlink*53]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*19] = sz[itlink+ntlink*55]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*20] = sz[itlink+ntlink*69]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*21] = sz[itlink+ntlink*64]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*22] = sz[itlink+ntlink*65]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*23] = sz[itlink+ntlink*68]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*24] = sz[itlink+ntlink*70]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*25] = sz[itlink+ntlink*84]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*26] = sz[itlink+ntlink*79]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*27] = sz[itlink+ntlink*80]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*28] = sz[itlink+ntlink*83]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*29] = sz[itlink+ntlink*85]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*30] = sz[itlink+ntlink*99]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*31] = sz[itlink+ntlink*94]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*32] = sz[itlink+ntlink*95]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*33] = sz[itlink+ntlink*98]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*34] = sz[itlink+ntlink*100]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*35] = sz[itlink+ntlink*114]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*36] = sz[itlink+ntlink*109]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*37] = sz[itlink+ntlink*110]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*38] = sz[itlink+ntlink*113]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*39] = sz[itlink+ntlink*115]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*40] = sz[itlink+ntlink*129]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*41] = sz[itlink+ntlink*124]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*42] = sz[itlink+ntlink*125]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*43] = sz[itlink+ntlink*128]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*44] = sz[itlink+ntlink*130]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*45] = sz[itlink+ntlink*144]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*46] = sz[itlink+ntlink*139]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*47] = sz[itlink+ntlink*140]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*48] = sz[itlink+ntlink*143]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*49] = sz[itlink+ntlink*145]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*50] = sz[itlink+ntlink*159]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*51] = sz[itlink+ntlink*154]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*52] = sz[itlink+ntlink*155]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*53] = sz[itlink+ntlink*158]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*54] = sz[itlink+ntlink*160]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*55] = sz[itlink+ntlink*174]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*56] = sz[itlink+ntlink*169]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*57] = sz[itlink+ntlink*170]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*58] = sz[itlink+ntlink*173]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*59] = sz[itlink+ntlink*175]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*60] = sz[itlink+ntlink*189]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*61] = sz[itlink+ntlink*184]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*62] = sz[itlink+ntlink*185]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*63] = sz[itlink+ntlink*188]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*64] = sz[itlink+ntlink*190]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*65] = sz[itlink+ntlink*204]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*66] = sz[itlink+ntlink*199]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*67] = sz[itlink+ntlink*200]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*68] = sz[itlink+ntlink*203]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*69] = sz[itlink+ntlink*205]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*70] = sz[itlink+ntlink*219]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*71] = sz[itlink+ntlink*214]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*72] = sz[itlink+ntlink*215]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*73] = sz[itlink+ntlink*218]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*74] = sz[itlink+ntlink*220]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*75] = sz[itlink+ntlink*234]/(log(1.0E+1)*(z[itlink+ntlink*9]+1.0));
  sy[it+nt*76] = sz[itlink+ntlink*229]/(log(1.0E+1)*(z[itlink+ntlink*4]+1.0));
  sy[it+nt*77] = sz[itlink+ntlink*230]/(log(1.0E+1)*(z[itlink+ntlink*5]+1.0));
  sy[it+nt*78] = sz[itlink+ntlink*233]/(log(1.0E+1)*(z[itlink+ntlink*8]+1.0));
  sy[it+nt*79] = sz[itlink+ntlink*235]/(log(1.0E+1)*(z[itlink+ntlink*10]+1.0));
  sy[it+nt*80] = 0.0;
  sy[it+nt*81] = 0.0;
  sy[it+nt*82] = 0.0;
  sy[it+nt*83] = 0.0;
  sy[it+nt*84] = 0.0;
  sy[it+nt*85] = 0.0;
  sy[it+nt*86] = 0.0;
  sy[it+nt*87] = 0.0;
  sy[it+nt*88] = 0.0;
  sy[it+nt*89] = 0.0;
  sy[it+nt*90] = 0.0;
  sy[it+nt*91] = 0.0;
  sy[it+nt*92] = 0.0;
  sy[it+nt*93] = 0.0;
  sy[it+nt*94] = 0.0;
  sy[it+nt*95] = 0.0;
  sy[it+nt*96] = 0.0;
  sy[it+nt*97] = 0.0;
  sy[it+nt*98] = 0.0;
  sy[it+nt*99] = 0.0;
  sy[it+nt*100] = 0.0;
  sy[it+nt*101] = 0.0;
  sy[it+nt*102] = 0.0;
  sy[it+nt*103] = 0.0;
  sy[it+nt*104] = 0.0;

  return;
}


 void fsystd_covid_data_5B6232583955DC35E6149014EA483B50(double t, int nt, int it, int ntlink, int itlink, double *systd, double *p, double *y, double *u, double *x, double *z, double *sy, double *su, double *sx, double *sz){
  systd[it+nt*0] = 0.0;
  systd[it+nt*1] = 0.0;
  systd[it+nt*2] = 0.0;
  systd[it+nt*3] = 0.0;
  systd[it+nt*4] = 0.0;
  systd[it+nt*5] = 0.0;
  systd[it+nt*6] = 0.0;
  systd[it+nt*7] = 0.0;
  systd[it+nt*8] = 0.0;
  systd[it+nt*9] = 0.0;
  systd[it+nt*10] = 0.0;
  systd[it+nt*11] = 0.0;
  systd[it+nt*12] = 0.0;
  systd[it+nt*13] = 0.0;
  systd[it+nt*14] = 0.0;
  systd[it+nt*15] = 0.0;
  systd[it+nt*16] = 0.0;
  systd[it+nt*17] = 0.0;
  systd[it+nt*18] = 0.0;
  systd[it+nt*19] = 0.0;
  systd[it+nt*20] = 0.0;
  systd[it+nt*21] = 0.0;
  systd[it+nt*22] = 0.0;
  systd[it+nt*23] = 0.0;
  systd[it+nt*24] = 0.0;
  systd[it+nt*25] = 0.0;
  systd[it+nt*26] = 0.0;
  systd[it+nt*27] = 0.0;
  systd[it+nt*28] = 0.0;
  systd[it+nt*29] = 0.0;
  systd[it+nt*30] = 0.0;
  systd[it+nt*31] = 0.0;
  systd[it+nt*32] = 0.0;
  systd[it+nt*33] = 0.0;
  systd[it+nt*34] = 0.0;
  systd[it+nt*35] = 0.0;
  systd[it+nt*36] = 0.0;
  systd[it+nt*37] = 0.0;
  systd[it+nt*38] = 0.0;
  systd[it+nt*39] = 0.0;
  systd[it+nt*40] = 0.0;
  systd[it+nt*41] = 0.0;
  systd[it+nt*42] = 0.0;
  systd[it+nt*43] = 0.0;
  systd[it+nt*44] = 0.0;
  systd[it+nt*45] = 0.0;
  systd[it+nt*46] = 0.0;
  systd[it+nt*47] = 0.0;
  systd[it+nt*48] = 0.0;
  systd[it+nt*49] = 0.0;
  systd[it+nt*50] = 0.0;
  systd[it+nt*51] = 0.0;
  systd[it+nt*52] = 0.0;
  systd[it+nt*53] = 0.0;
  systd[it+nt*54] = 0.0;
  systd[it+nt*55] = 0.0;
  systd[it+nt*56] = 0.0;
  systd[it+nt*57] = 0.0;
  systd[it+nt*58] = 0.0;
  systd[it+nt*59] = 0.0;
  systd[it+nt*60] = 0.0;
  systd[it+nt*61] = 0.0;
  systd[it+nt*62] = 0.0;
  systd[it+nt*63] = 0.0;
  systd[it+nt*64] = 0.0;
  systd[it+nt*65] = 0.0;
  systd[it+nt*66] = 0.0;
  systd[it+nt*67] = 0.0;
  systd[it+nt*68] = 0.0;
  systd[it+nt*69] = 0.0;
  systd[it+nt*70] = 0.0;
  systd[it+nt*71] = 0.0;
  systd[it+nt*72] = 0.0;
  systd[it+nt*73] = 0.0;
  systd[it+nt*74] = 0.0;
  systd[it+nt*75] = 0.0;
  systd[it+nt*76] = 0.0;
  systd[it+nt*77] = 0.0;
  systd[it+nt*78] = 0.0;
  systd[it+nt*79] = 0.0;
  systd[it+nt*80] = 1.0;
  systd[it+nt*81] = 0.0;
  systd[it+nt*82] = 0.0;
  systd[it+nt*83] = 0.0;
  systd[it+nt*84] = 0.0;
  systd[it+nt*85] = 0.0;
  systd[it+nt*86] = 0.0;
  systd[it+nt*87] = 0.0;
  systd[it+nt*88] = 1.0;
  systd[it+nt*89] = 0.0;
  systd[it+nt*90] = 0.0;
  systd[it+nt*91] = 1.0;
  systd[it+nt*92] = 0.0;
  systd[it+nt*93] = 0.0;
  systd[it+nt*94] = 0.0;
  systd[it+nt*95] = 0.0;
  systd[it+nt*96] = 0.0;
  systd[it+nt*97] = 1.0;
  systd[it+nt*98] = 0.0;
  systd[it+nt*99] = 0.0;
  systd[it+nt*100] = 0.0;
  systd[it+nt*101] = 0.0;
  systd[it+nt*102] = 0.0;
  systd[it+nt*103] = 0.0;
  systd[it+nt*104] = 1.0;

  return;
}


 void fy_scale_covid_data_5B6232583955DC35E6149014EA483B50(double t, int nt, int it, int ntlink, int itlink, int ny, int nx, int nz, int iruns, double *y_scale, double *p, double *u, double *x, double *z, double *dfzdx){
  y_scale[ny*nt*iruns+it+nt*0] = dfzdx[nx*ntlink*iruns+itlink+ntlink*9]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*9]+1.0));
  y_scale[ny*nt*iruns+it+nt*1] = dfzdx[nx*ntlink*iruns+itlink+ntlink*4]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*4]+1.0));
  y_scale[ny*nt*iruns+it+nt*2] = dfzdx[nx*ntlink*iruns+itlink+ntlink*5]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*5]+1.0));
  y_scale[ny*nt*iruns+it+nt*3] = dfzdx[nx*ntlink*iruns+itlink+ntlink*8]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*8]+1.0));
  y_scale[ny*nt*iruns+it+nt*4] = dfzdx[nx*ntlink*iruns+itlink+ntlink*10]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*10]+1.0));
  y_scale[ny*nt*iruns+it+nt*5] = dfzdx[nx*ntlink*iruns+itlink+ntlink*24]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*9]+1.0));
  y_scale[ny*nt*iruns+it+nt*6] = dfzdx[nx*ntlink*iruns+itlink+ntlink*19]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*4]+1.0));
  y_scale[ny*nt*iruns+it+nt*7] = dfzdx[nx*ntlink*iruns+itlink+ntlink*20]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*5]+1.0));
  y_scale[ny*nt*iruns+it+nt*8] = dfzdx[nx*ntlink*iruns+itlink+ntlink*23]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*8]+1.0));
  y_scale[ny*nt*iruns+it+nt*9] = dfzdx[nx*ntlink*iruns+itlink+ntlink*25]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*10]+1.0));
  y_scale[ny*nt*iruns+it+nt*10] = dfzdx[nx*ntlink*iruns+itlink+ntlink*39]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*9]+1.0));
  y_scale[ny*nt*iruns+it+nt*11] = dfzdx[nx*ntlink*iruns+itlink+ntlink*34]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*4]+1.0));
  y_scale[ny*nt*iruns+it+nt*12] = dfzdx[nx*ntlink*iruns+itlink+ntlink*35]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*5]+1.0));
  y_scale[ny*nt*iruns+it+nt*13] = dfzdx[nx*ntlink*iruns+itlink+ntlink*38]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*8]+1.0));
  y_scale[ny*nt*iruns+it+nt*14] = dfzdx[nx*ntlink*iruns+itlink+ntlink*40]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*10]+1.0));
  y_scale[ny*nt*iruns+it+nt*15] = dfzdx[nx*ntlink*iruns+itlink+ntlink*54]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*9]+1.0));
  y_scale[ny*nt*iruns+it+nt*16] = dfzdx[nx*ntlink*iruns+itlink+ntlink*49]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*4]+1.0));
  y_scale[ny*nt*iruns+it+nt*17] = dfzdx[nx*ntlink*iruns+itlink+ntlink*50]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*5]+1.0));
  y_scale[ny*nt*iruns+it+nt*18] = dfzdx[nx*ntlink*iruns+itlink+ntlink*53]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*8]+1.0));
  y_scale[ny*nt*iruns+it+nt*19] = dfzdx[nx*ntlink*iruns+itlink+ntlink*55]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*10]+1.0));
  y_scale[ny*nt*iruns+it+nt*20] = dfzdx[nx*ntlink*iruns+itlink+ntlink*69]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*9]+1.0));
  y_scale[ny*nt*iruns+it+nt*21] = dfzdx[nx*ntlink*iruns+itlink+ntlink*64]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*4]+1.0));
  y_scale[ny*nt*iruns+it+nt*22] = dfzdx[nx*ntlink*iruns+itlink+ntlink*65]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*5]+1.0));
  y_scale[ny*nt*iruns+it+nt*23] = dfzdx[nx*ntlink*iruns+itlink+ntlink*68]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*8]+1.0));
  y_scale[ny*nt*iruns+it+nt*24] = dfzdx[nx*ntlink*iruns+itlink+ntlink*70]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*10]+1.0));
  y_scale[ny*nt*iruns+it+nt*25] = dfzdx[nx*ntlink*iruns+itlink+ntlink*84]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*9]+1.0));
  y_scale[ny*nt*iruns+it+nt*26] = dfzdx[nx*ntlink*iruns+itlink+ntlink*79]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*4]+1.0));
  y_scale[ny*nt*iruns+it+nt*27] = dfzdx[nx*ntlink*iruns+itlink+ntlink*80]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*5]+1.0));
  y_scale[ny*nt*iruns+it+nt*28] = dfzdx[nx*ntlink*iruns+itlink+ntlink*83]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*8]+1.0));
  y_scale[ny*nt*iruns+it+nt*29] = dfzdx[nx*ntlink*iruns+itlink+ntlink*85]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*10]+1.0));
  y_scale[ny*nt*iruns+it+nt*30] = dfzdx[nx*ntlink*iruns+itlink+ntlink*99]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*9]+1.0));
  y_scale[ny*nt*iruns+it+nt*31] = dfzdx[nx*ntlink*iruns+itlink+ntlink*94]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*4]+1.0));
  y_scale[ny*nt*iruns+it+nt*32] = dfzdx[nx*ntlink*iruns+itlink+ntlink*95]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*5]+1.0));
  y_scale[ny*nt*iruns+it+nt*33] = dfzdx[nx*ntlink*iruns+itlink+ntlink*98]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*8]+1.0));
  y_scale[ny*nt*iruns+it+nt*34] = dfzdx[nx*ntlink*iruns+itlink+ntlink*100]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*10]+1.0));
  y_scale[ny*nt*iruns+it+nt*35] = dfzdx[nx*ntlink*iruns+itlink+ntlink*114]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*9]+1.0));
  y_scale[ny*nt*iruns+it+nt*36] = dfzdx[nx*ntlink*iruns+itlink+ntlink*109]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*4]+1.0));
  y_scale[ny*nt*iruns+it+nt*37] = dfzdx[nx*ntlink*iruns+itlink+ntlink*110]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*5]+1.0));
  y_scale[ny*nt*iruns+it+nt*38] = dfzdx[nx*ntlink*iruns+itlink+ntlink*113]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*8]+1.0));
  y_scale[ny*nt*iruns+it+nt*39] = dfzdx[nx*ntlink*iruns+itlink+ntlink*115]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*10]+1.0));
  y_scale[ny*nt*iruns+it+nt*40] = dfzdx[nx*ntlink*iruns+itlink+ntlink*129]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*9]+1.0));
  y_scale[ny*nt*iruns+it+nt*41] = dfzdx[nx*ntlink*iruns+itlink+ntlink*124]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*4]+1.0));
  y_scale[ny*nt*iruns+it+nt*42] = dfzdx[nx*ntlink*iruns+itlink+ntlink*125]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*5]+1.0));
  y_scale[ny*nt*iruns+it+nt*43] = dfzdx[nx*ntlink*iruns+itlink+ntlink*128]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*8]+1.0));
  y_scale[ny*nt*iruns+it+nt*44] = dfzdx[nx*ntlink*iruns+itlink+ntlink*130]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*10]+1.0));
  y_scale[ny*nt*iruns+it+nt*45] = dfzdx[nx*ntlink*iruns+itlink+ntlink*144]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*9]+1.0));
  y_scale[ny*nt*iruns+it+nt*46] = dfzdx[nx*ntlink*iruns+itlink+ntlink*139]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*4]+1.0));
  y_scale[ny*nt*iruns+it+nt*47] = dfzdx[nx*ntlink*iruns+itlink+ntlink*140]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*5]+1.0));
  y_scale[ny*nt*iruns+it+nt*48] = dfzdx[nx*ntlink*iruns+itlink+ntlink*143]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*8]+1.0));
  y_scale[ny*nt*iruns+it+nt*49] = dfzdx[nx*ntlink*iruns+itlink+ntlink*145]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*10]+1.0));
  y_scale[ny*nt*iruns+it+nt*50] = dfzdx[nx*ntlink*iruns+itlink+ntlink*159]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*9]+1.0));
  y_scale[ny*nt*iruns+it+nt*51] = dfzdx[nx*ntlink*iruns+itlink+ntlink*154]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*4]+1.0));
  y_scale[ny*nt*iruns+it+nt*52] = dfzdx[nx*ntlink*iruns+itlink+ntlink*155]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*5]+1.0));
  y_scale[ny*nt*iruns+it+nt*53] = dfzdx[nx*ntlink*iruns+itlink+ntlink*158]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*8]+1.0));
  y_scale[ny*nt*iruns+it+nt*54] = dfzdx[nx*ntlink*iruns+itlink+ntlink*160]/(log(1.0E+1)*(z[nz*ntlink*iruns+itlink+ntlink*10]+1.0));

  return;
}


