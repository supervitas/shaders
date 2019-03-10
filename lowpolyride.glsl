// Author: supervitas


#ifdef GL_ES
precision highp float;
#endif



#define AA 1
#define MAX_MARCHING_STEPS 255
#define MIN_DIST 0.0 // near
#define MAX_DIST  100. // far
#define EPSILON 0.0001
#define PI 3.1415926535



#define TREE_LEAVES vec3(0.091,0.260,0.082)
#define TREE_LEAVES_YELLOW vec3(0.359,0.485,0.121)
#define TRUNK vec3(0.175,0.050,0.005)

#define CAR_BODY vec3(255,173,0)
#define CAR_ROOF vec3(0.625,0.282,0.086)
#define CAR_GLASS vec3(0.460,0.480,0.450)
#define CAR_TOP_BAG vec3(0.302,0.877,0.960)
#define CAR_TIRES vec3(0.060,0.060,0.060)

#define GREEN_GRASS vec3(0.137,0.645,0.163)
#define ROAD vec3(0.150,0.150,0.150)

#define ROAD_WIDTH 12.752


// gradient background
#define BACK_COL_TOP vec3(0.000,0.579,0.825)
#define BACK_COL_BOTTOM vec3(0.149,0.244,0.785)


uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

float random (in vec2 _st) {
    return fract(sin(dot(_st.xy,
                         vec2(12.9898,78.233)))*
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


mat2 terrainProps = mat2(0.1,-0.1, 0.1,0.1);



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

float sdTorus( vec3 p, vec2 t ) {
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

vec4 intersectSDF(vec4 d1, vec4 d2) {
	return (d1.x>d2.x) ? d1 : d2;
}

vec4 unionSDF(vec4 d1, vec4 d2) {
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

vec4 tree1(vec3 p) {	
  vec4 trunc = vec4(sdCappedCylinder(rotateX(PI) * p + vec3(0.,-.5,0) , vec2(0.2, 2.)), TRUNK);
  vec4 leaf = vec4(sdOctahedron(p + vec3(0., -3, 0.), 2.120), TREE_LEAVES);
                       
  return unionSDF(trunc, leaf);
}

vec4 tree2(vec3 p) {
    vec4 trunc = vec4(sdCappedCylinder(rotateX(PI) * p + vec3(0., -.5, 0) , vec2(0.2,1.990)), TRUNK);
	vec4 leaf = vec4(sdBox(p + vec3(0., -4, 0.), vec3(1.2)), TREE_LEAVES);
    
  leaf = unionSDF(leaf, vec4(sdBox(rotateX(1.904 * PI) * rotateZ(.04 * PI) * p + vec3(0., -4, 0.), vec3(1.2)), TREE_LEAVES));

  return unionSDF(trunc, leaf);
}

vec4 tree3(vec3 p) {	
  vec4 trunc = vec4(sdCappedCylinder(rotateX(PI) * p + vec3(0., -.5, 0) , vec2(0.2,2.990)), TRUNK);
  vec4 leaf = vec4(sdHexPrism(p + vec3(0, -3.5, 0.), vec2(1.70, 1.)), TREE_LEAVES);

  return unionSDF(trunc, leaf);
}

vec4 tree4(vec3 p) {	
  vec4 trunc = vec4(sdCappedCylinder(rotateX(PI) * p + vec3(0., -1.5, 0) , vec2(0.2,4.0)), TRUNK);
  vec4 leaf = vec4(sdOctahedron(p + vec3(0., -4, 0.), 2.120), TREE_LEAVES);
  leaf = unionSDF(leaf, vec4(sdOctahedron(p + vec3(0., -5.7, 0.), 1.3), TREE_LEAVES));
  leaf = unionSDF(leaf, vec4(sdOctahedron(p + vec3(0., -6.8, 0.), 0.7), TREE_LEAVES));
  

  return unionSDF(trunc, leaf);
}


vec4 tree5(vec3 p) {
	vec4 trunc = vec4(sdCappedCylinder(rotateX(PI) * p + vec3(0., -1.5, 0) , vec2(0.2,4.0)), TRUNK);
	vec4 leaf = vec4(sdBox(p + vec3(0., -4, 0.), vec3(1.5)), TREE_LEAVES);
	leaf = unionSDF(leaf, vec4(sdBox(p + vec3(1.2, -3.5, 1.5), vec3(0.8)), TREE_LEAVES_YELLOW));
    leaf = unionSDF(leaf, vec4(sdBox(p + vec3(-0.9, -5.5, -1.2), vec3(0.9)), TREE_LEAVES_YELLOW));
    

	return unionSDF(trunc, leaf);
}

vec4 createTrees(vec3 samplePoint) {
    vec4 scene = vec4(1.);
    
    
	const float zMax = -30.;
    const float zMin = 15.;
    
    float z = fract(u_time);
    

	vec4 tree1 = tree1(samplePoint + vec3(15.2, -2.5,  mix(zMax,  zMin, mod(u_time, 1.5))));
    vec4 tree2 = tree2(samplePoint + vec3(18.4, -2.2, mix(zMax, zMin, mod(2.5 + u_time, 1.5))));
    vec4 tree3 = tree3(samplePoint + vec3(22.2, -3.2, mix(zMax, zMin, mod(2.7 + u_time, 1.5))));
    vec4 tree4 = tree4(samplePoint + vec3(22.2, -2.2, mix(zMax, zMin, mod(0.2 + u_time, 1.5))));
    vec4 tree5 = tree5(samplePoint + vec3(20.2, -2, mix(zMax, zMin, mod(0.8 + u_time, 1.5))));

    scene = tree1;


 
    scene = unionSDF(scene, tree2);
    scene = unionSDF(scene, tree3);
    scene = unionSDF(scene, tree4);
    scene = unionSDF(scene, tree5);
    
    
    
    return scene;
}


vec4 createCar(vec3 p) {
    float jumping = mix(-0.2, -0.3, abs(sin(u_time * 4.)));
    const float height = 0.5;


    vec4 body = vec4(sdBox(p + vec3(0., jumping - height, 0), vec3(1.35, height, 2.5)), CAR_BODY);
    vec4 roof = vec4(sdBox(p + vec3(0., jumping - 2.5, 0), vec3(1.35, 0.1, 1)), CAR_ROOF);
    vec4 windowBack = vec4(sdBox(rotateX(2.100) * p + vec3(0.,  2.1, 1.2), vec3(1.1, 0.1, 0.65)), CAR_GLASS);
    
    vec4 car = unionSDF(body, roof);
    car = unionSDF(car, windowBack);
    
    vec4 bagazh = vec4(sdCappedCylinder(rotateZ(PI ) * rotateX(3.726)  * p + vec3(1.22, -1.15 , -2.1), vec2(0.13, 0.75)), TRUNK);
     car = unionSDF(car, bagazh);
    
    vec4 bagazh2 = vec4(sdCappedCylinder(rotateZ(PI ) * rotateX(3.726)  * p + vec3(-1.22, -1.15, -2.1), vec2(0.13, 0.75)), TRUNK);
     car = unionSDF(car, bagazh2);
    
    // for (float i = 0.; i < 3.; i++) {
    //      vec2 holder = vec2(sdCappedCylinder( rotateZ(PI )  * rotateX( PI / 2.) * p + vec3(0.7 - 0.7 * i , 0., 2.85 + abs(jumping)) , vec2(0.1, 0.5)), 14.0);
    //      car = unionSDF(car, holder);
    // }


    vec3 t = rotateZ(PI / 2.) * rotateX(PI / 2.) * p;
   
    vec4 wheel = vec4(sdTorus(t + (vec3(1.5, 1.52, .1) ), vec2(0.5,0.2) ), CAR_TIRES);
    vec4 wheel2 = vec4(sdTorus(t + vec3(-1.5, 1.52, .1), vec2(0.5,0.2)), CAR_TIRES);
    vec4 wheel3 = vec4(sdTorus(t + vec3(-1.5, -1.52, .1), vec2(0.5,0.2)), CAR_TIRES);
    vec4 wheel4 = vec4(sdTorus(t + vec3(1.5, -1.52, .1), vec2(0.5,0.2)), CAR_TIRES);
    
    car = unionSDF(car, wheel);
    car = unionSDF(car, wheel2);
    car = unionSDF(car, wheel3);
    car = unionSDF(car, wheel4);
    
    return car;
}


vec4 map(vec3 samplePoint) {
    vec4 scene;

    
    vec4 plane = vec4(sdPlane(samplePoint), ROAD);
    if (mod(samplePoint.z + 16. * u_time, 16.) > 6.600 && samplePoint.x < 0. && samplePoint.x > -0.7) {
        plane.yzw = vec3(1.0);
    }
    
    vec4 trees = vec4(1.0);
    
    if (samplePoint.x < -ROAD_WIDTH || samplePoint.x > ROAD_WIDTH ) {
        trees = createTrees(samplePoint);
        
//         vec2 p = 1.5 * samplePoint.xz;
//         p.y += 1.1 * u_time;
//         vec3 noise = noise(p);
//         plane.yzw = vec3(noise.x);
        
//          plane.yzw = mix(GREEN_GRASS, vec3(0.392,0.700,0.084), noise.x);
        
        plane.yzw = GREEN_GRASS;
    }


    vec4 car = createCar(samplePoint + vec3(6., -1.5, -2.5));

    scene = unionSDF(car, plane);
	scene = unionSDF(scene, plane);
    
    scene = unionSDF(trees, scene);
    
    return scene;
}

vec4 raymarsh(vec3 eye, vec3 marchingDirection, float start, float end) {
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        vec4 dist = map(eye + depth * marchingDirection);
        if (dist.x < EPSILON) {
			return vec4(depth, dist.yzw);
        }
        depth += dist.x;
        if (depth >= end) {
            return vec4(end);
        }
    }

    return vec4(end);
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


vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye) {
    const vec3 ambientLight = .8 * vec3(1.0, 1.0, 1.0);
    vec3 color = ambientLight * k_a;
    
    vec3 light1Pos = vec3(4.0, 12.0,4.0);
    vec3 light1Intensity = vec3(0.885,0.885,0.885);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye,light1Pos,light1Intensity);
      
    return color;
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


float calcAO( vec3 pos, vec3 nor ) {
	float occ = 0.0;
    float sca = 1.0;
    for(int i=0; i < 5; i++ ) {
        float hr = 0.01 + 0.12*float(i)/4.0;
        vec3 aopos =  nor * hr + pos;
        float dd = map( aopos ).x;
        occ += -(dd-hr)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 );    
}

vec3 render(vec2 p, vec2 uv) {
  // vec3 ro = vec3(mix(-2., 2., sin(u_time)), 5., -8.);
      vec3 ro = vec3(5., 25., -10.6);
     // ro.z += 20. * u_time;
    // vec2 m = u_mouse / u_resolution;
    // vec3 ro = 9.0*normalize(vec3(sin(3.0*m.x), 2.4*m.y, cos(3.0*m.x)));
    
    vec3 ta =  normalize(vec3(0., -0.1, -1.000));
    mat3 ca = calcLookAtMatrix(ro, ta, 0.0);
    vec3 rd = ca * normalize(vec3(p.xy, 1.2));
    
    vec3 color = vec3(1.0);
    vec4 scene = raymarsh(ro, rd, MIN_DIST, MAX_DIST);
    
    if (scene.x > MAX_DIST - EPSILON) { // background
        color = background(uv);
    } else {
       color = scene.yzw;
        
         vec3 p = ro + scene.x * rd;
         vec3 nor = getNormal(p);
     

        float shininess = 1.0;
    	color *= phongIllumination(vec3(2.5), vec3(1.5), vec3(0.5), shininess, p, ro);
		color *= calcAO( p, nor );

    }
    
	return color;
}


void main() {
     vec2 uv = gl_FragCoord.xy / u_resolution;
#if AA>1
    vec3 color = vec3(0.0);
    for( int m=0; m<AA; m++ )
    for( int n=0; n<AA; n++ ) {
        vec2 px = gl_FragCoord.xy + vec2(float(m),float(n)) / float(AA);
        vec2 p = (-u_resolution.xy+2.0*px) / u_resolution.y;
    	color += render( p, uv );    
    }
    color /= float(AA*AA);
#else
 	vec2 p = (-u_resolution.xy + 2.0*gl_FragCoord.xy) / u_resolution.y;
    vec3 color = render(p, uv);
#endif 
 
    // color *= 0.25+0.334*pow( 16.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.3 ); // Vigneting
	// color = pow(color, vec3(1. / 2.2)); // gamma correction
    // color = smoothstep(0., 1., color);
    gl_FragColor = vec4(color, 1.0);
}