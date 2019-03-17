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



#define TREE_LEAVES vec3(0.154,0.970,0.153)
#define TREE_LEAVES_YELLOW vec3(0.359,0.485,0.121)
#define TRUNK vec3(0.175,0.050,0.005)

#define CAR_BODY vec3(255,173,0)
#define CAR_ROOF vec3(0.625,0.282,0.086)
#define CAR_GLASS vec3(0.388,0.405,0.380)
#define CAR_TOP_BAG vec3(0.302,0.877,0.960)
#define CAR_TIRES vec3(0.060,0.060,0.060)

#define GREEN_GRASS vec3(0.133,0.175,0.154)
#define ROAD vec3(0.150,0.150,0.150)

#define ROAD_WIDTH 12.752
#define TREES_ROAD_OFFSET_RIGHT ROAD_WIDTH + 2.


// gradient background
#define BACK_COL_TOP vec3(0.000,0.579,0.825)
#define BACK_COL_BOTTOM vec3(0.149,0.244,0.785)

#define SPEED 26.


uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;


float random (in vec2 _st) {
    return fract(sin(dot(_st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
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

vec4 opSmoothUnion( vec4 d1, vec4 d2, float k ) {
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

vec4 tree1(vec3 p, float randValue) {
  vec4 trunc = vec4(sdCappedCylinder(((rotateX(PI) * p + vec3(0.,-.5,0)) / randValue) * randValue, vec2(0.2, 2.) * randValue) , TRUNK);
  vec4 leaf = vec4(sdOctahedron(((rotateY(PI * randValue) * p + vec3(0., -3. * randValue, 0.)) / randValue) , 2.120) * randValue, TREE_LEAVES);
                       
  return unionSDF(trunc, leaf);
}

vec4 tree2(vec3 p, float randValue) {
  vec4 trunc = vec4(sdCappedCylinder(((rotateX(PI) * p + vec3(0., -.5, 0)) / randValue) * randValue, vec2(0.2,2.990) * randValue), TRUNK);
  vec4 leaf = vec4(sdHexPrism(((rotateY(PI / randValue) * p + vec3(0, -3.5 * randValue, 0.)) / randValue) * randValue, vec2(1.75* randValue, 1.2 * randValue / 1.5) ), vec3(0.113,0.500,0.373));

  return unionSDF(trunc, leaf);
}

vec4 tree3(vec3 p, float randValue) {
  vec4 trunc = vec4(sdCappedCylinder(((rotateX(PI) * p + vec3(0., -1.5, 0) / randValue) * randValue) , vec2(0.2,4.0) * randValue), TRUNK);
  vec4 leaf = vec4(sdOctahedron(p + vec3(0., -3, 0.), 2.120), TREE_LEAVES);
  leaf = unionSDF(leaf, vec4(sdOctahedron(p + vec3(0., -5.7, 0.), 1.3), TREE_LEAVES));
  leaf = unionSDF(leaf, vec4(sdOctahedron(p + vec3(0., -6.8, 0.), 0.7), TREE_LEAVES));
  

  return unionSDF(trunc, leaf);
}


vec4 tree4(vec3 p, float randValue) {
	vec4 trunc = vec4(sdCappedCylinder(((rotateX(PI) * p + vec3(0., -1.5, 0)) / randValue) * randValue , vec2(0.2,4.0) * randValue), TRUNK);
	vec4 leaf = vec4(sdBox((( p + vec3(0., -4, 0.)) / randValue) * randValue, vec3(1.5) * randValue), TREE_LEAVES);
	leaf = unionSDF(leaf, vec4(sdBox((( p + vec3(1.2, -3.5,2.5)) / randValue) * randValue, vec3(0.8, 0.5, 0.6) * randValue), TREE_LEAVES_YELLOW));
    leaf = unionSDF(leaf, vec4(sdBox((( p + vec3(-1.9 , -5.2 - randValue , -1. - randValue)) / randValue) * randValue, vec3(0.8, 0.5, 0.9) * randValue), TREE_LEAVES_YELLOW));

	return unionSDF(trunc, leaf);
}

vec2 getRandByIndex(float i) {
    float rand = random(vec2(i));
    float randVal = rand + i  + u_time * SPEED * 0.02;
    
    return vec2(rand, randVal);
}


vec4 createTrees(vec3 samplePoint) {
    vec4 scene = vec4(1.);
    
    
	const float zMax = -35.;
    const float zMin = 15.;
    const float rowWidth = 10.;

	for (float i = 0.; i < 2.; i ++) {
       	vec4 tree = tree1(samplePoint + vec3(TREES_ROAD_OFFSET_RIGHT + rowWidth * i, -2.5,  mix(zMax,  zMin, mod(getRandByIndex(i).y , 1.))), 1.); 
        vec4 tree2 = tree2(samplePoint + vec3(TREES_ROAD_OFFSET_RIGHT + rowWidth * i , -2.5,  mix(zMax,  zMin, mod(getRandByIndex(i + 5.).y, 1.))), 1.); 
        vec4 tree3 = tree3(samplePoint + vec3(TREES_ROAD_OFFSET_RIGHT + rowWidth * i, -2.5,  mix(zMax,  zMin, mod(getRandByIndex(i + 10.).y, 1.))), 1.); 
        vec4 tree4 = tree4(samplePoint + vec3(TREES_ROAD_OFFSET_RIGHT + rowWidth * i, -2.5,  mix(zMax,  zMin, mod(getRandByIndex(i + 15.).y, 1.))), 1.); 

        
        scene = unionSDF(scene, tree);
        // scene = unionSDF(scene, tree2);
        // scene = unionSDF(scene, tree3);
        // scene = unionSDF(scene, tree4);
    }

    
//     for (float i = 1.; i < 3.; i++) {
//         float step = i * 105.;
//         float rand = random(vec2(step));
//         float randVal = rand + 1.5 + step + u_time;
//         float rand2 = random(vec2(ceil(randVal)));

//        	vec4 tree = tree2(samplePoint + vec3(TREES_ROAD_OFFSET_RIGHT +  rand2 * 20., -2.5,  mix(zMax,  zMin, mod(randVal , 1.))), 0.8 + rand * rand2); 

        
//         scene = unionSDF(scene, tree);
//     }
    
//      for (float i = 1.; i < 3.; i++) {
//         float step = i  * 2890.;
//         float rand = random(vec2(step));
//         float randVal = rand + 1.5 + step + u_time;
//         float rand2 = random(vec2(ceil(randVal)));

//        	vec4 tree = tree3(samplePoint + vec3(TREES_ROAD_OFFSET_RIGHT +  rand2 * 15., -2.5,  mix(zMax,  zMin, mod(randVal , 1.))), 0.9 + rand * rand2); 

        
//         scene = unionSDF(scene, tree);
//     }
    
//     for (float i = 1.; i < 3.; i++) {
//         float step = i  * 1492.5;
//         float rand = random(vec2(step));
//         float randVal = rand + 1.5 + step + u_time;
//         float rand2 = random(vec2(ceil(randVal)));

//        	vec4 tree = tree4(samplePoint + vec3(TREES_ROAD_OFFSET_RIGHT +  rand2 * 25., -2.5,  mix(zMax,  zMin, mod(randVal , 1.))), 0.8 + rand * rand2); 

        
//         scene = unionSDF(scene, tree);
//     }


    
    return scene;
}


vec4 createCar(vec3 p) {
    float jumping = mix(-0.2, -0.4, abs(sin(u_time * 4.)));
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

vec4 createFence(vec3 p) {
    vec4 pillar = vec4(sdBox(p + vec3(TREES_ROAD_OFFSET_RIGHT - 2., -.5, 0), vec3(.15, 2., 100.)), vec3(0.315,0.180,0.110));
	vec4 fence = vec4(sdBox(p + vec3(TREES_ROAD_OFFSET_RIGHT - 2., -2.5, 0), vec3(.25, 0.12, 100.)), vec3(0.300,0.219,0.090));
    
    vec4 pillarLeft = vec4(sdBox(p + vec3(-TREES_ROAD_OFFSET_RIGHT - 2., -.5, 0), vec3(.15, 2., 100.)), vec3(0.315,0.180,0.110));
	vec4 fenceLeft = vec4(sdBox(p + vec3(-TREES_ROAD_OFFSET_RIGHT - 2., -2.5, 0), vec3(.25, 0.12, 100.)), vec3(0.300,0.219,0.090));

   if (mod(p.z + SPEED * u_time, SPEED) > .5) {
       pillar.x = 0.35;
       pillarLeft.x = 0.35;
    }
    
    vec4 fenceR = unionSDF(pillar, fence);
    vec4 fenceL = unionSDF(pillarLeft, fenceLeft);
    
    return  unionSDF(fenceL, fenceR);
}

vec4 map(vec3 samplePoint) {
    vec4 plane = vec4(sdPlane(samplePoint), ROAD);
    if (mod(samplePoint.z + SPEED * u_time , 16.) > 6.600 && samplePoint.x < 0. && samplePoint.x > -0.7) {
        plane.yzw = vec3(1.0);
    }
    
    vec4 trees = createTrees(samplePoint);

    if (samplePoint.x < -ROAD_WIDTH || samplePoint.x > ROAD_WIDTH ) {
        // vec4 grass = texture2D(grass, gl_FragCoord.xy / u_resolution);
        plane.yzw = GREEN_GRASS;
        // plane.yzw = grass.rgb;
    }

    vec4 car = createCar(samplePoint + vec3(6., -1.5, -2.5));
    vec4 fence = createFence(samplePoint);
    
    vec4 scene = unionSDF(car, plane);
    scene = unionSDF(scene, fence);
    scene = unionSDF(scene, trees);
    
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


float calcSoftshadow(in vec3 ro, in vec3 rd) {
	float res = 1.0;
    float t = 0.01;
    for( int i=0; i<256; i++ ) {
		float h = map( ro + rd*t ).x;
        res = min( res, smoothstep(0.0,1.0,18.0*h/t) );
        t += clamp( h, 0.005, 0.02 );
        if( res < .5 || t > 0.3 ) break;
    }
    return clamp( res, 0.0, 1.0 );
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


vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye, vec3 lp) {
    const vec3 ambientLight = .8 * vec3(1.0, 1.0, 1.0);
    vec3 color = ambientLight * k_a;
    
    vec3 light1Pos = lp;
    vec3 light1Intensity = vec3(1.0);
    
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
     

        float shininess = 5.824;

        color *= phongIllumination(vec3(1.5), vec3(1.5), vec3(1.1), shininess, p , ro, vec3(0.0, 205.0, 0.0));
        // color *= calcSoftshadow(p, ro);
		// color *= calcAO( p, nor );

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
 
    color *= 0.25+0.334*pow( 16.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.3 ); // Vigneting
	// color = pow(color, vec3(1. / 2.2)); // gamma correction
    // color = smoothstep(0., 1., color);
    gl_FragColor = vec4(color, 1.0);
}
