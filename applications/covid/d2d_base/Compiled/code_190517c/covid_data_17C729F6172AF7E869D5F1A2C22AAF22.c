#include "covid_data_17C729F6172AF7E869D5F1A2C22AAF22.h"
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





 void fy_covid_data_17C729F6172AF7E869D5F1A2C22AAF22(double t, int nt, int it, int ntlink, int itlink, int ny, int nx, int nz, int iruns, double *y, double *p, double *u, double *x, double *z){
  y[ny*nt*iruns+it+nt*0] = log(x[nx*ntlink*iruns+itlink+ntlink*9]+1.0)/log(1.0E+1);
  y[ny*nt*iruns+it+nt*1] = log(x[nx*ntlink*iruns+itlink+ntlink*4]+1.0)/log(1.0E+1);
  y[ny*nt*iruns+it+nt*2] = log(x[nx*ntlink*iruns+itlink+ntlink*5]+1.0)/log(1.0E+1);
  y[ny*nt*iruns+it+nt*3] = log(x[nx*ntlink*iruns+itlink+ntlink*8]+1.0)/log(1.0E+1);
  y[ny*nt*iruns+it+nt*4] = log(x[nx*ntlink*iruns+itlink+ntlink*10]+1.0)/log(1.0E+1);

  return;
}


 void fystd_covid_data_17C729F6172AF7E869D5F1A2C22AAF22(double t, int nt, int it, int ntlink, int itlink, double *ystd, double *y, double *p, double *u, double *x, double *z){
  ystd[it+nt*0] = p[16];
  ystd[it+nt*1] = p[18];
  ystd[it+nt*2] = p[19];
  ystd[it+nt*3] = p[17];
  ystd[it+nt*4] = p[20];

  return;
}


 void fsy_covid_data_17C729F6172AF7E869D5F1A2C22AAF22(double t, int nt, int it, int ntlink, int itlink, double *sy, double *p, double *u, double *x, double *z, double *su, double *sx, double *sz){
  sy[it+nt*0] = sx[itlink+ntlink*9]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*1] = sx[itlink+ntlink*4]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*2] = sx[itlink+ntlink*5]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*3] = sx[itlink+ntlink*8]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*4] = sx[itlink+ntlink*10]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*5] = sx[itlink+ntlink*20]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*6] = sx[itlink+ntlink*15]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*7] = sx[itlink+ntlink*16]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*8] = sx[itlink+ntlink*19]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*9] = sx[itlink+ntlink*21]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*10] = sx[itlink+ntlink*31]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*11] = sx[itlink+ntlink*26]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*12] = sx[itlink+ntlink*27]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*13] = sx[itlink+ntlink*30]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*14] = sx[itlink+ntlink*32]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*15] = sx[itlink+ntlink*42]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*16] = sx[itlink+ntlink*37]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*17] = sx[itlink+ntlink*38]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*18] = sx[itlink+ntlink*41]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*19] = sx[itlink+ntlink*43]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*20] = sx[itlink+ntlink*53]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*21] = sx[itlink+ntlink*48]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*22] = sx[itlink+ntlink*49]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*23] = sx[itlink+ntlink*52]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*24] = sx[itlink+ntlink*54]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*25] = sx[itlink+ntlink*64]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*26] = sx[itlink+ntlink*59]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*27] = sx[itlink+ntlink*60]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*28] = sx[itlink+ntlink*63]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*29] = sx[itlink+ntlink*65]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*30] = sx[itlink+ntlink*75]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*31] = sx[itlink+ntlink*70]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*32] = sx[itlink+ntlink*71]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*33] = sx[itlink+ntlink*74]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*34] = sx[itlink+ntlink*76]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*35] = sx[itlink+ntlink*86]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*36] = sx[itlink+ntlink*81]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*37] = sx[itlink+ntlink*82]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*38] = sx[itlink+ntlink*85]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*39] = sx[itlink+ntlink*87]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*40] = sx[itlink+ntlink*97]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*41] = sx[itlink+ntlink*92]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*42] = sx[itlink+ntlink*93]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*43] = sx[itlink+ntlink*96]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*44] = sx[itlink+ntlink*98]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*45] = sx[itlink+ntlink*108]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*46] = sx[itlink+ntlink*103]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*47] = sx[itlink+ntlink*104]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*48] = sx[itlink+ntlink*107]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*49] = sx[itlink+ntlink*109]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*50] = sx[itlink+ntlink*119]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*51] = sx[itlink+ntlink*114]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*52] = sx[itlink+ntlink*115]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*53] = sx[itlink+ntlink*118]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*54] = sx[itlink+ntlink*120]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*55] = sx[itlink+ntlink*130]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*56] = sx[itlink+ntlink*125]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*57] = sx[itlink+ntlink*126]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*58] = sx[itlink+ntlink*129]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*59] = sx[itlink+ntlink*131]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*60] = sx[itlink+ntlink*141]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*61] = sx[itlink+ntlink*136]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*62] = sx[itlink+ntlink*137]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*63] = sx[itlink+ntlink*140]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*64] = sx[itlink+ntlink*142]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*65] = sx[itlink+ntlink*152]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*66] = sx[itlink+ntlink*147]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*67] = sx[itlink+ntlink*148]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*68] = sx[itlink+ntlink*151]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*69] = sx[itlink+ntlink*153]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*70] = sx[itlink+ntlink*163]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*71] = sx[itlink+ntlink*158]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*72] = sx[itlink+ntlink*159]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*73] = sx[itlink+ntlink*162]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*74] = sx[itlink+ntlink*164]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
  sy[it+nt*75] = sx[itlink+ntlink*174]/(log(1.0E+1)*(x[itlink+ntlink*9]+1.0));
  sy[it+nt*76] = sx[itlink+ntlink*169]/(log(1.0E+1)*(x[itlink+ntlink*4]+1.0));
  sy[it+nt*77] = sx[itlink+ntlink*170]/(log(1.0E+1)*(x[itlink+ntlink*5]+1.0));
  sy[it+nt*78] = sx[itlink+ntlink*173]/(log(1.0E+1)*(x[itlink+ntlink*8]+1.0));
  sy[it+nt*79] = sx[itlink+ntlink*175]/(log(1.0E+1)*(x[itlink+ntlink*10]+1.0));
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


 void fsystd_covid_data_17C729F6172AF7E869D5F1A2C22AAF22(double t, int nt, int it, int ntlink, int itlink, double *systd, double *p, double *y, double *u, double *x, double *z, double *sy, double *su, double *sx, double *sz){
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


 void fy_scale_covid_data_17C729F6172AF7E869D5F1A2C22AAF22(double t, int nt, int it, int ntlink, int itlink, int ny, int nx, int nz, int iruns, double *y_scale, double *p, double *u, double *x, double *z, double *dfzdx){
  y_scale[ny*nt*iruns+it+nt*0] = 0.0;
  y_scale[ny*nt*iruns+it+nt*1] = 0.0;
  y_scale[ny*nt*iruns+it+nt*2] = 0.0;
  y_scale[ny*nt*iruns+it+nt*3] = 0.0;
  y_scale[ny*nt*iruns+it+nt*4] = 0.0;
  y_scale[ny*nt*iruns+it+nt*5] = 0.0;
  y_scale[ny*nt*iruns+it+nt*6] = 0.0;
  y_scale[ny*nt*iruns+it+nt*7] = 0.0;
  y_scale[ny*nt*iruns+it+nt*8] = 0.0;
  y_scale[ny*nt*iruns+it+nt*9] = 0.0;
  y_scale[ny*nt*iruns+it+nt*10] = 0.0;
  y_scale[ny*nt*iruns+it+nt*11] = 0.0;
  y_scale[ny*nt*iruns+it+nt*12] = 0.0;
  y_scale[ny*nt*iruns+it+nt*13] = 0.0;
  y_scale[ny*nt*iruns+it+nt*14] = 0.0;
  y_scale[ny*nt*iruns+it+nt*15] = 0.0;
  y_scale[ny*nt*iruns+it+nt*16] = 0.0;
  y_scale[ny*nt*iruns+it+nt*17] = 0.0;
  y_scale[ny*nt*iruns+it+nt*18] = 0.0;
  y_scale[ny*nt*iruns+it+nt*19] = 0.0;
  y_scale[ny*nt*iruns+it+nt*20] = 0.0;
  y_scale[ny*nt*iruns+it+nt*21] = 1.0/(log(1.0E+1)*(x[nx*ntlink*iruns+itlink+ntlink*4]+1.0));
  y_scale[ny*nt*iruns+it+nt*22] = 0.0;
  y_scale[ny*nt*iruns+it+nt*23] = 0.0;
  y_scale[ny*nt*iruns+it+nt*24] = 0.0;
  y_scale[ny*nt*iruns+it+nt*25] = 0.0;
  y_scale[ny*nt*iruns+it+nt*26] = 0.0;
  y_scale[ny*nt*iruns+it+nt*27] = 1.0/(log(1.0E+1)*(x[nx*ntlink*iruns+itlink+ntlink*5]+1.0));
  y_scale[ny*nt*iruns+it+nt*28] = 0.0;
  y_scale[ny*nt*iruns+it+nt*29] = 0.0;
  y_scale[ny*nt*iruns+it+nt*30] = 0.0;
  y_scale[ny*nt*iruns+it+nt*31] = 0.0;
  y_scale[ny*nt*iruns+it+nt*32] = 0.0;
  y_scale[ny*nt*iruns+it+nt*33] = 0.0;
  y_scale[ny*nt*iruns+it+nt*34] = 0.0;
  y_scale[ny*nt*iruns+it+nt*35] = 0.0;
  y_scale[ny*nt*iruns+it+nt*36] = 0.0;
  y_scale[ny*nt*iruns+it+nt*37] = 0.0;
  y_scale[ny*nt*iruns+it+nt*38] = 0.0;
  y_scale[ny*nt*iruns+it+nt*39] = 0.0;
  y_scale[ny*nt*iruns+it+nt*40] = 0.0;
  y_scale[ny*nt*iruns+it+nt*41] = 0.0;
  y_scale[ny*nt*iruns+it+nt*42] = 0.0;
  y_scale[ny*nt*iruns+it+nt*43] = 1.0/(log(1.0E+1)*(x[nx*ntlink*iruns+itlink+ntlink*8]+1.0));
  y_scale[ny*nt*iruns+it+nt*44] = 0.0;
  y_scale[ny*nt*iruns+it+nt*45] = 1.0/(log(1.0E+1)*(x[nx*ntlink*iruns+itlink+ntlink*9]+1.0));
  y_scale[ny*nt*iruns+it+nt*46] = 0.0;
  y_scale[ny*nt*iruns+it+nt*47] = 0.0;
  y_scale[ny*nt*iruns+it+nt*48] = 0.0;
  y_scale[ny*nt*iruns+it+nt*49] = 0.0;
  y_scale[ny*nt*iruns+it+nt*50] = 0.0;
  y_scale[ny*nt*iruns+it+nt*51] = 0.0;
  y_scale[ny*nt*iruns+it+nt*52] = 0.0;
  y_scale[ny*nt*iruns+it+nt*53] = 0.0;
  y_scale[ny*nt*iruns+it+nt*54] = 1.0/(log(1.0E+1)*(x[nx*ntlink*iruns+itlink+ntlink*10]+1.0));

  return;
}


