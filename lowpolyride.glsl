// Author: supervitas


#ifdef GL_ES
precision highp float;
#endif

#define MAX_MARCHING_STEPS 255
#define MIN_DIST 0.0 // near
#define MAX_DIST  100. // far
#define EPSILON 0.0001
#define PI 3.1415926535


// gradient background
#define BACK_COL_TOP vec3(0.000,0.579,0.825)
#define BACK_COL_BOTTOM vec3(0.149,0.244,0.785)

uniform sampler2D u_tex0; // https://images-na.ssl-images-amazon.com/images/I/910PPWWqFuL.png
uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;



float random (in vec2 _st) {
    return fract(sin(dot(_st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
}

// Based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
float noise (in vec2 _st) {
    vec2 i = floor(_st);
    vec2 f = fract(_st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

#define NUM_OCTAVES 10

float fbm ( in vec2 _st) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(140.0);
    // Rotate to reduce axial bias
    mat2 rot = mat2(cos(0.5), sin(0.5),
                    -sin(0.5), cos(0.50));
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * noise(_st);
        _st = rot * _st * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

vec3 background(vec2 uv) {
    return mix(BACK_COL_TOP, BACK_COL_BOTTOM, uv.y);
}



mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}

mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

mat3 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, -s, 0),
        vec3(s, c, 0),
        vec3(0, 0, 1)
    );
}

float sdSphere( vec3 p, float s ) {
  return length(p)-s;
}

float sdBox( vec3 p, vec3 b ) {
  vec3 d = abs(p) - b;
  return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf
}

float sdCappedCylinder( vec3 p, vec2 h ) {
  vec2 d = abs(vec2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdTriPrism(vec3 p, vec2 h) {
    vec3 q = abs(p);
    return max(q.z-h.y,max(q.x*0.866025+p.y*0.5,-p.y)-h.x*0.5);
}

float sdOctahedron(in vec3 p, in float s) {
    p = abs(p);
    return (p.x+p.y+p.z-s)*0.57735027;
}

float sdHexPrism( vec3 p, vec2 h ) {
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0*min(dot(k.xy, p.xy), 0.)*k.xy;
    vec2 d = vec2(length(p.xy-vec2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x),
       p.z-h.y );

    return min(max(d.x,d.y), 0.0) + length(max(d, 0.0));
}


float sdPlane( vec3 p ) {
	return p.y;
}

vec2 intersectSDF(vec2 d1, vec2 d2) {
	return (d1.x>d2.x) ? d1 : d2;
}

vec2 unionSDF(vec2 d1, vec2 d2) {
    return (d1.x<d2.x) ? d1 : d2;
}

float opRepSphere( vec3 p, vec3 c, float radius) {
    vec3 q = mod(p,c)-0.5*c;
    return sdSphere( q, radius);
}

vec2 opSmoothUnion( vec2 d1, vec2 d2, float k ) {
    float h = clamp( 0.5 + 0.5*(d2.x - d1.x) / k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h);
}

vec2 opSmoothSubtraction( vec2 d1, vec2 d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2.x + d1.x) /k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h);
}

vec2 opSmoothIntersection( vec2 d1, vec2 d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2.x - d1.x) / k, 0.0, 1.0 );
    return mix( d2, d1, h ) + k * h * (1.0 - h);
}


float differenceSDF(float distA, float distB) {
    return max(distA, -distB);
}

vec2 tree1(vec3 p) {
  vec2 trunc = vec2(sdCappedCylinder(rotateX(PI) * p + vec3(0.,-.5,0) , vec2(0.2, 2.)), 2.0);
  vec2 leaf = vec2(sdOctahedron(p + vec3(0., -3, 0.), 2.120), 1.0);

  return unionSDF(trunc, leaf);
}

vec2 tree2(vec3 p) {
  vec2 trunc = vec2(sdCappedCylinder(rotateX(PI) * p + vec3(0., -.5, 0) , vec2(0.2,1.990)), 2.0);
  vec2 leaf = vec2(sdTriPrism(p + vec3(0, -2, 0.), vec2(1.70, 1.)), 1.0);

  return unionSDF(trunc, leaf);
}

vec2 tree3(vec3 p) {
  vec2 trunc = vec2(sdCappedCylinder(rotateX(PI) * p + vec3(0., -.5, 0) , vec2(0.2,1.990)), 2.0);
  vec2 leaf = vec2(sdHexPrism(p + vec3(0, -2, 0.), vec2(1.70, 1.)), 1.0);

  return unionSDF(trunc, leaf);
}

vec2 tree4(vec3 p) {
  vec2 trunc = vec2(sdCappedCylinder(rotateX(PI) * p + vec3(0., -.5, 0) , vec2(0.2,1.990)), 2.0);
  vec2 leaf = vec2(sdOctahedron(p + vec3(0., -3, 0.), 2.120), 1.0);
  leaf = unionSDF(leaf, vec2(sdOctahedron(p + vec3(0., -4.7, 0.), 1.3), 1.0));
  leaf = unionSDF(leaf, vec2(sdOctahedron(p + vec3(0., -6., 0.), 0.7), 1.0));


  return unionSDF(trunc, leaf);
}


vec2 map(vec3 samplePoint) { // vec2.y - is ID
    vec2 scene;

    vec2 plane = vec2(sdPlane(samplePoint + vec3(0, 3.0, 0.)), 3.);
    // vec2 tower = tower(samplePoint + vec3(0, 1.2, 3.));
	vec2 tree1 = tree1(samplePoint + vec3(-2.2, 1.2, 3.));
    vec2 tree2 = tree2(samplePoint + vec3(2.2, 1.2, 3.));
    vec2 tree3 = tree3(samplePoint + vec3(-2.2, 1.2, -5.));
    vec2 tree4 = tree4(samplePoint + vec3(2.2, 1.2, -5.));

    scene = unionSDF(tree1, plane);
    scene = unionSDF(scene, tree2);
    scene = unionSDF(scene, tree2);
    scene = unionSDF(scene, tree3);
    scene = unionSDF(scene, tree4);


	scene = unionSDF(scene, plane);


    return scene;
}

vec2 raymarsh(vec3 eye, vec3 marchingDirection, float start, float end) {
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        vec2 dist = map(eye + depth * marchingDirection);
        if (dist.x < EPSILON) {
			return vec2(depth, dist.y);
        }
        depth += dist.x;
        if (depth >= end) {
            return vec2(end);
        }
    }

    return vec2(end);
}

vec3 getNormal(vec3 p) {
    return normalize(vec3(
        map(vec3(p.x + EPSILON, p.y, p.z)).x - map(vec3(p.x - EPSILON, p.y, p.z)).x,
        map(vec3(p.x, p.y + EPSILON, p.z)).x - map(vec3(p.x, p.y - EPSILON, p.z)).x,
        map(vec3(p.x, p.y, p.z  + EPSILON)).x - map(vec3(p.x, p.y, p.z - EPSILON)).x
    ));
}


vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye, vec3 lightPos, vec3 lightIntensity) {
    vec3 N = getNormal(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));

    float dotLN = dot(L, N);
    float dotRV = dot(R, V);

    if (dotLN < 0.0) { // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    }

    if (dotRV < 0.0) { // Light reflection in opposite direction as viewer, apply only diffuse component
        return lightIntensity * (k_d * dotLN);
    }
    return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

struct light {
    vec3 lightPosition;

    vec3 amibnetColor;
    float ambientIntencity;

    vec3 directLightColor;
	vec3 directLightIntencity;

    vec3 specularLightColor;
    float specularLightIntencity;
};

vec3 phongIllumination(light data, vec3 p, vec3 eye) {
    vec3 ambientColor = data.ambientIntencity * data.amibnetColor;
	vec3 phongColor = phongContribForLight(data.directLightColor, data.specularLightColor, data.specularLightIntencity, p, eye, data.lightPosition, data.directLightIntencity);

    return ambientColor + phongColor;
}


vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

mat3 viewMatrix(vec3 eye, vec3 center, vec3 up) {
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    return mat3(s, u, -f);
}

mat3 calcLookAtMatrix(vec3 origin, vec3 target, float roll) {
  vec3 rr = vec3(sin(roll), cos(roll), 0.0);
  vec3 ww = normalize(target - origin);
  vec3 uu = normalize(cross(ww, rr));
  vec3 vv = normalize(cross(uu, ww));

  return mat3(uu, vv, ww);
}

float softShadow(vec3 ro, vec3 rd, float tmin, float tmax, float k) {
    float res = 1.0;
    float t = tmin;
    for( int i = 0; i < 150; i++ ) {
		float h = map( ro + rd*t ).x;
        res = min( res, 8.0 * h / t );
        t += clamp( h, 0.02, 0.10 );
        if( res<0.005 || t > tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

float calcAO( vec3 pos, vec3 nor ) {
	float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i < 5; i++ ) {
        float hr = 0.01 + 0.12*float(i)/4.0;
        vec3 aopos =  nor * hr + pos;
        float dd = map( aopos ).x;
        occ += -(dd-hr)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 );
}


void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 p = (-u_resolution.xy + 2.0*gl_FragCoord.xy)/u_resolution.y;

    const float speed = 0.5;
    // vec3 ro = vec3( 2.5 * cos(speed * u_time + 6.0), 0.5, 2.5 * sin(speed * u_time + 6.0) );

    vec3 ro = vec3(mix(-2., 2., (sin(u_time))), 5., -10.);

    vec3 ta = vec3(0,0, -1.000);
    mat3 ca = calcLookAtMatrix(ro, ta, 0.0);
    vec3 rd = ca * normalize(vec3(p.xy, 1.2));

    vec3 color = vec3(1.0);
    vec2 scene = raymarsh(ro, rd, MIN_DIST, MAX_DIST);

    if (scene.x > MAX_DIST - EPSILON) { // background
        color = background(uv);
    } else {
        light light1 = light(
        vec3(6.000,14.617,3.794), // light position
        vec3(0.735,0.745,0.737), 0.620, // ambient color - ambient intencity
        vec3(0.885,0.831,0.839), vec3(3.), // direct light color - direct light intencity
        vec3(0.910,0.861,0.879), 3.260); // specular color  - specular power

         if (scene.y >= 1. && scene.y <= 2.) {
             color = vec3(0.268,0.695,0.318);
         }

         if (scene.y >= 2. && scene.y <= 3.) {
             color = vec3(0.130,0.037,0.004);
         }

    	 if (scene.y >= 3. && scene.y <= 4.) {
             color = vec3(0.545,0.545,0.545);
         }

         vec3 p = ro + scene.x * rd;
         vec3 nor = getNormal(p);
         vec3 ref = reflect( rd, nor );

       	 color *= phongIllumination(light1, p, ro);
	     // color *= softShadow(p, light1.lightPosition, 0.492, 2.5, 32.);
		 color *= calcAO( p, nor );

    }

    color *= 0.25+0.334*pow( 16.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.3 ); // Vigneting
	color = pow(color, vec3(1. / 2.2)); // gamma correction
    gl_FragColor = vec4(color,1.0);
}
