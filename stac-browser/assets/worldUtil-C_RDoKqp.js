import{bp as a}from"./GeoJSON-DhAITP2G.js";function p(r,s){const n=r.viewState.projection,t=s.getSource().getWrapX()&&n.canWrapX(),o=n.getExtent(),c=r.extent,e=t?a(o):null,l=t?Math.ceil((c[2]-o[2])/e)+1:1;return[t?Math.floor((c[0]-o[0])/e):0,l,e]}export{p as g};
//# sourceMappingURL=worldUtil-C_RDoKqp.js.map
