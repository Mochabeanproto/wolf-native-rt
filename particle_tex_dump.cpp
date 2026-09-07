#define NOMINMAX
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "shared/bridge.h"

static void decode565(uint16_t c,uint8_t* o){o[2]=(uint8_t)(((c>>11)&31)*255/31);o[1]=(uint8_t)(((c>>5)&63)*255/63);o[0]=(uint8_t)((c&31)*255/31);o[3]=255;}
int main(){
 HANDLE m=OpenFileMappingA(FILE_MAP_READ,FALSE,BRIDGE_SHM_NAME); if(!m)return 2;
 void* b=MapViewOfFile(m,FILE_MAP_READ,0,0,(SIZE_T)bridge_total_size()); if(!b)return 3;
 const BridgeHeader* h=(const BridgeHeader*)b; const BridgeTex* tt=bridge_texs(b); const uint8_t* arena=bridge_arena(b);
 const unsigned ids[]={59,60,61,62,63};
 for(unsigned id:ids){ if(id>=h->texCount||tt[id].fmt!=BTEX_BC3)continue; const BridgeTex&t=tt[id];
  std::vector<uint8_t> px((size_t)t.width*t.height*4); const uint8_t* src=arena+t.arenaOff;
  for(uint32_t by=0;by<(t.height+3)/4;++by)for(uint32_t bx=0;bx<(t.width+3)/4;++bx){const uint8_t*q=src+by*t.rowPitch+bx*16;
   uint8_t ap[8]={q[0],q[1]}; if(ap[0]>ap[1])for(int i=1;i<=6;++i)ap[i+1]=(uint8_t)(((7-i)*ap[0]+i*ap[1])/7);else{for(int i=1;i<=4;++i)ap[i+1]=(uint8_t)(((5-i)*ap[0]+i*ap[1])/5);ap[6]=0;ap[7]=255;}
   uint64_t ai=0;for(int i=0;i<6;++i)ai|=(uint64_t)q[2+i]<<(8*i);
   uint16_t c0=q[8]|(q[9]<<8),c1=q[10]|(q[11]<<8);uint8_t cp[4][4];decode565(c0,cp[0]);decode565(c1,cp[1]);
   for(int k=0;k<3;++k){cp[2][k]=(uint8_t)((2*cp[0][k]+cp[1][k])/3);cp[3][k]=(uint8_t)((cp[0][k]+2*cp[1][k])/3);}cp[2][3]=cp[3][3]=255;
   uint32_t ci=q[12]|(q[13]<<8)|(q[14]<<16)|(q[15]<<24);
   for(int y=0;y<4;++y)for(int x=0;x<4;++x){uint32_t X=bx*4+x,Y=by*4+y;if(X>=t.width||Y>=t.height)continue;int n=y*4+x,cc=(ci>>(2*n))&3,aa=(ai>>(3*n))&7;uint8_t*o=&px[((size_t)Y*t.width+X)*4];o[0]=cp[cc][0];o[1]=cp[cc][1];o[2]=cp[cc][2];o[3]=ap[aa];}
  }
  char fn[128];sprintf_s(fn,"build\\particle_tex_%u.tga",id);FILE*f=nullptr;fopen_s(&f,fn,"wb");if(!f)continue;uint8_t hd[18]={};hd[2]=2;hd[12]=t.width&255;hd[13]=t.width>>8;hd[14]=t.height&255;hd[15]=t.height>>8;hd[16]=32;hd[17]=0x28;fwrite(hd,1,18,f);fwrite(px.data(),1,px.size(),f);fclose(f);printf("%s\n",fn);
 }
 UnmapViewOfFile(b);CloseHandle(m);return 0;
}
