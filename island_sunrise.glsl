// Author: supervitas

#ifdef GL_ES
precision mediump float;
#endif


uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

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

  vec2 df = 1.0*f*f*(f*(f-2.0)+1.0);
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
    vec3 n = noise(p);
    vec2 df = n.yz;
    float f = abs(0.5 * n.x / (2.5 + dot(df, df)));
    return f;
}

float sdOctahedron(in vec3 p, in float s) {
    p = abs(p);
    return (p.x+p.y+p.z-s)*0.57735027;
}

float sdBox( vec3 p, vec3 b ) {
  vec3 d = abs(p) - b;
  return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
}

float sdSphere( vec3 p, float s ) {
  return length(p)-s;
}

mat2 terrainProps = mat2(0.620,-0.952, 0.868,0.008);
float fbmIs(vec2 p) {
  vec2 df = vec2(0.0);
  float f = 0.0;
  float w = .5;

  for (int i = 0; i < 4; i++) {
    vec3 n = noise(p);
    df += n.yz;
    f += abs(w * n.x / (1.0 + dot(df, df)));
    w *= 0.5;
    p = 2. * terrainProps * p;
  }
  return f;
}

vec4 islands(vec3 p) {
    float fbmNoise = fbmIs(p.xz);
    vec3 position = p + vec3(5.422, -1.8,-7.5);
    float sphere = sdSphere(position, 3.5);
    float result = fbmNoise + sphere;
    
    vec4 islands = vec4(result, mix(vec3(0.086,0.335,0.062), vec3(0.745,0.558,0.178),  step(position.y, 0.500)));
    return islands;
}

vec4 unionSDF(vec4 d1, vec4 d2) {
    return (d1.x<d2.x) ? d1 : d2;
}


vec4 map(vec3 p) {
    vec4 scene = vec4(p.y, 0.0,0.0,0.0);
    vec4 color = vec4(0.0);
    
    vec2 pointOverTime = p.xz + u_time;
    float h = fbmL(pointOverTime);

    // vec3 nor = normal(point, scene);

    scene.x -= h;
    scene.yzw = mix(vec3(0.094,0.372,0.440), vec3(1.0), h);
    
    scene = unionSDF(scene, islands(p));

  	return scene;
}

vec4 raymarsh(vec3 eye, vec3 marchingDirection) {
    float depth = 0.0;

    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        vec4 dist = map(eye + depth * marchingDirection);
        if (dist.x < EPSILON) {
			return vec4(depth, dist.yzw);
        }
        depth += dist.x;
        if (depth >= MAX_DIST) {
            return vec4(-1, vec3(0.0));
        }
    }

    return vec4(-1, vec3(0.0));
}

vec3 normal(vec3 pos, float t) {
	vec2  eps = vec2( 0.002*t, 0.0 );
    return normalize( vec3( fbmL(pos.xz-eps.xy) - fbmL(pos.xz+eps.xy),
                            2.0*eps.x,
                            fbmL(pos.xz-eps.yx) - fbmL(pos.xz+eps.yx) ) );
}


vec3 calcLights(vec3 p, vec3 eye, vec3 N) {
  vec3 L = normalize(vec3(5., 55.0,  50.));
 vec3 light = vec3(1.0) * max(dot(N, L), 0.);

  return light;
}

mat3 calcLookAtMatrix(vec3 origin, vec3 target, float roll) {
  vec3 rr = vec3(sin(roll), cos(roll), 0.0);
  vec3 ww = normalize(target - origin);
  vec3 uu = normalize(cross(ww, rr));
  vec3 vv = normalize(cross(uu, ww));

  return mat3(uu, vv, ww);
}


float getSun(vec2 uv) {
      vec2 dist = uv -vec2(0.640,0.750);
    const float radius = 0.02;
    float isCircle = 1.-smoothstep(radius-(radius*0.6),
                         radius+(radius*.4),
                         dot(dist,dist)*4.0);
    
	return isCircle;
}

void setSkyColor(out vec3 color, vec3 dir) {
    vec2 uv =  gl_FragCoord.xy/u_resolution.xy;
   color = mix(vec3(0.040,0.057,0.095), vec3(0.440,0.830,0.822), uv.y);
   float sun = getSun(uv);
   color = mix(color, vec3(0.965,0.807,0.096), sun);
}

vec3 trace(vec3 ro, vec3 rd) {
  vec3 color = vec3(0.0);
    
  vec4 scene = raymarsh(ro, rd);
  vec3 point = ro + scene.x * rd;
  if (scene.x > -1.) {
    vec3 nor = normal(point, scene.x);
      color = scene.yzw;

    color *= calcLights(point, ro, nor);
  } else {
    setSkyColor(color, rd);
  }
    
 return color;
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

  vec3 color = trace(ro, rd);

  color = pow(color, vec3(1. / 2.2)); // gamma correction
  color = smoothstep(0., 1.,color);

  gl_FragColor = vec4(color,1.0);
}
