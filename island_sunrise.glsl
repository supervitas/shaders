// Author @patriciogv - 2015
// http://patriciogonzalezvivo.com

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

// Author: supervitas

#define MAX_MARCHING_STEPS 256
#define MAX_DIST 25. // far
#define EPSILON 0.001
#define PI 3.1415926535


float random( in vec2 _st) {
  return fract(sin(dot(_st.xy,
      vec2(12.9898, 78.233))) *
    43758.5453123);
}

vec3 noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);

  vec2 df = 20.0*f*f*(f*(f-2.0)+1.0);
  f = f*f*f*(f*(f*6.-15.)+10.);

  float a = random(i + vec2(0.5));
  float b = random(i + vec2(1.5, 0.5));
  float c = random(i + vec2(.5, 1.5));
  float d = random(i + vec2(1.5, 1.5));

  float k = a - b - c + d;
  float n = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);

  return vec3(n, vec2(b - a + k * f.y, c - a + k * f.x) * df);
}


float fbmL(vec2 p) {
  vec2 df = vec2(0.0);
  float f = 0.0;
  float w = 0.5;

  for (int i = 0; i < 2; i++) {
    vec3 n = noise(p);
    df += n.yz;
    f += abs(w * n.x / (1.0 + dot(df, df)));
    w *= 0.5;
    p = 2. * 0.1 * p;
  }
  return f;
}



float map(vec3 p) {
    float scene = p.y;
    
    vec2 pointOverTime = p.xz + u_time;

    float h = fbmL(pointOverTime);
    scene -= h;

  	return scene;
}

float raymarch(vec3 ro, vec3 rd) {
  float d = 0.;
  float t = 0.;
  for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
    d = map(ro + t * rd);
    if (d < EPSILON * t || t > MAX_DIST) break;
    t += 0.5 * d;
  }

  return d < EPSILON * t ? t : -1.;
}

vec3 normal(vec3 pos, float t) {
	vec2  eps = vec2( 0.002*t, 0.0 );
    return normalize( vec3( fbmL(pos.xz-eps.xy) - fbmL(pos.xz+eps.xy),
                            2.0*eps.x,
                            fbmL(pos.xz-eps.yx) - fbmL(pos.xz+eps.yx) ) );
}

struct light {
  vec3 lightPosition;
  vec3 amibnetColor;
  float ambientIntencity;
  vec3 directLightColor;
  vec3 directLightIntencity;
};

vec3 diffuseLight(vec3 k_d, vec3 p, vec3 eye, vec3 lightPos, vec3 lightIntensity) {
  vec3 N = normal(p, 0.01);
  vec3 L = normalize(lightPos - p);

  float dotLN = dot(L, N);

  if (dotLN < 0.0) {
    return vec3(0.0, 0.0, 0.0);
  }

  return lightIntensity * (k_d * dotLN);
}

vec3 calcLights(light data, vec3 p, vec3 eye) {
  vec3 ambientColor = data.ambientIntencity * data.amibnetColor;
  vec3 phongColor = diffuseLight(data.directLightColor, p, eye, data.lightPosition, data.directLightIntencity);

  return ambientColor + phongColor;
}

mat3 calcLookAtMatrix(vec3 origin, vec3 target, float roll) {
  vec3 rr = vec3(sin(roll), cos(roll), 0.0);
  vec3 ww = normalize(target - origin);
  vec3 uu = normalize(cross(ww, rr));
  vec3 vv = normalize(cross(uu, ww));

  return mat3(uu, vv, ww);
}

void setColor(vec3 p, vec3 n, out vec3 color) {
  color = vec3(0.098,0.237,0.865);
}

float getSun(vec2 uv) {
      vec2 dist = uv -vec2(0.640,0.750);
    const float radius = 0.02;
    float isCircle = 1.-smoothstep(radius-(radius*0.01),
                         radius+(radius*0.01),
                         dot(dist,dist)*4.0);
    
	return isCircle;
}

void setSkyColor(vec2 uv, out vec3 color, vec3 dir) {
   color = mix(vec3(0.040,0.057,0.095), vec3(0.440,0.830,0.822), uv.y);
   float sun = getSun(uv);
   color = mix(color, vec3(0.965,0.807,0.096), sun);
}

void main() {
  vec2 uv =  gl_FragCoord.xy/u_resolution.xy;
  vec2 p = (-u_resolution.xy + 2.0 * gl_FragCoord.xy) / u_resolution.y;

  float speed = 0.002;
  float terrainEndTime = abs(sin(u_time * speed));
  vec3 ro = vec3(0.,1.8,1.);


  vec3 target = ro + vec3(0., 0., 0.1); // revert camera when near to end
  mat3 cam = calcLookAtMatrix(ro, target, 0.);
  vec3 rd = cam * normalize(vec3(p.xy, 1.0));

  vec3 color = vec3(0.0);
  float scene = raymarch(ro, rd);
  vec3 point = ro + scene * rd;
  if (scene > -1.) {
    light light1 = light(
      ro + vec3(10., 150., 100.), // light position
      vec3(0.931,0.975,0.906), 0.412, // ambient color - ambient intencity
      vec3(0.254,1.000,0.777), vec3(0.162,0.555,0.560)); // direct light color - direct light intencity


    vec3 nor = normal(point, scene);

    setColor(point, nor, color);

    color *= calcLights(light1, point, ro);
  } else {
    point = ro + scene * rd;
    setSkyColor(uv, color, rd);
  }

  color = pow(color, vec3(1. / 2.2)); // gamma correction
  color = smoothstep(0., 1.,color);

  gl_FragColor = vec4(color,1.0);
}
