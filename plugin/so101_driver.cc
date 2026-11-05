// Contact-operated screwdriver. No policy control channel or attachment constraint.
#include <mujoco/mujoco.h>
#include <mujoco/mjplugin.h>
#include <algorithm>
#include <cstdint>
#include <cmath>

namespace {
struct Driver { int forward_q, reverse_q, spindle_v; };
const char* attrs[] = {"forward", "reverse", "spindle"};
double box(const mjtNum* p, double hx, double hy, double hz, double cz) {
  double x=std::abs(p[0])-hx, y=std::abs(p[1])-hy, z=std::abs(p[2]-cz)-hz;
  return std::sqrt(std::pow(std::max(x,0.),2)+std::pow(std::max(y,0.),2)+std::pow(std::max(z,0.),2))
    +std::min(std::max({x,y,z}),0.);
}
double cylinder(const mjtNum* p, double r, double z0, double z1) {
  double a=std::hypot(p[0],p[1])-r, b=std::abs(p[2]-(z0+z1)/2)-(z1-z0)/2;
  return std::hypot(std::max(a,0.),std::max(b,0.))+std::min(std::max(a,b),0.);
}
double head(const mjtNum* p) {
  double solid=std::min(cylinder(p,.0035,0,.0004),cylinder(p,.0025,.0004,.0014));
  double recess=std::min(box(p,.0016,.0004,.001,.0018),box(p,.0004,.0016,.001,.0018));
  return std::max(solid,-recess);
}
}

mjPLUGIN_LIB_INIT(so101_driver) {
  mjpPlugin p;
  mjp_defaultPlugin(&p);
  p.name = "so101.contact_driver";
  p.capabilityflags = mjPLUGIN_PASSIVE;
  p.needstage = mjSTAGE_VEL;
  p.nattribute = 3;
  p.attributes = attrs;
  p.nstate = +[](const mjModel*, int) { return 0; };
  p.init = +[](const mjModel* m, mjData* d, int instance) {
    int j[3];
    for (int i=0; i<3; ++i) {
      j[i] = mj_name2id(m, mjOBJ_JOINT, mj_getPluginConfig(m, instance, attrs[i]));
      if (j[i] < 0) return -1;
    }
    auto* v = new Driver{m->jnt_qposadr[j[0]], m->jnt_qposadr[j[1]], m->jnt_dofadr[j[2]]};
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(v);
    return 0;
  };
  p.destroy = +[](mjData* d, int i) { delete reinterpret_cast<Driver*>(d->plugin_data[i]); d->plugin_data[i]=0; };
  p.reset = +[](const mjModel*, mjtNum*, void*, int) {};
  p.compute = +[](const mjModel*, mjData* d, int i, int) {
    const auto* v = reinterpret_cast<Driver*>(d->plugin_data[i]);
    const bool f = d->qpos[v->forward_q] > .0006;
    const bool r = d->qpos[v->reverse_q] > .0006;
    if (f == r) return;  // neither pressed or conflicting commands
    // Nominal low-torque setting; 250 rpm no-load. Dynamics need hardware identification.
    const double target = (f ? 1 : -1) * 26.17993878;
    d->qfrc_passive[v->spindle_v] += std::clamp(.002 * (target-d->qvel[v->spindle_v]), -.05, .05);
  };
  mjp_registerPlugin(&p);
  mjpPlugin h;
  mjp_defaultPlugin(&h);
  h.name="so101.screw_head";
  h.capabilityflags=mjPLUGIN_SDF;
  h.nstate=+[](const mjModel*,int){return 0;};
  h.init=+[](const mjModel*,mjData*,int){return 0;};
  h.reset=+[](const mjModel*,mjtNum*,void*,int){};
  h.compute=+[](const mjModel*,mjData*,int,int){};
  h.sdf_distance=+[](const mjtNum p[3],const mjData*,int){return head(p);};
  h.sdf_staticdistance=+[](const mjtNum p[3],const mjtNum*){return head(p);};
  h.sdf_gradient=+[](mjtNum g[3],const mjtNum p[3],const mjData*,int){
    for(int i=0;i<3;i++){mjtNum q[3]={p[0],p[1],p[2]};q[i]+=1e-7;g[i]=(head(q)-head(p))/1e-7;}
  };
  h.sdf_attribute=+[](mjtNum*,const char*[],const char*[]){};
  h.sdf_aabb=+[](mjtNum b[6],const mjtNum*){b[0]=b[1]=0;b[2]=.0007;b[3]=b[4]=.0036;b[5]=.0008;};
  mjp_registerPlugin(&h);
}
