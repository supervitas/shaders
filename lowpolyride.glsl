// Author: supervitas


#ifdef GL_ES
precision highp float;
#endif

#define AA 1
#define MAX_MARCHING_STEPS 255
#define MIN_DIST 0.0 // near
#define MAX_DIST  150. // far
#define EPSILON 0.001
#define PI 3.1415926535

#define TRUNK vec3(0.175,0.050,0.005)
#define CAR_TIRES vec3(0.060,0.060,0.060)
#define ROAD vec3(0.150,0.150,0.150)
#define ROAD_WIDTH 12.752
#define TREES_ROAD_OFFSET_RIGHT ROAD_WIDTH + 2.
#define SPEED 26.


uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;


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


float sdBox( vec3 p, vec3 b ) {
  vec3 d = abs(p) - b;
  return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf 
}

float piramidSDF(vec3 p, vec3 size) {
    vec3 ap = abs(p);
    vec3 d = ap - size;
    return max(dot(normalize(size), d), -p.y);
}

float sdCappedCylinder( vec3 p, vec2 h ) {
  vec2 d = abs(vec2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
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


vec4 unionSDF(vec4 d1, vec4 d2) {
    return (d1.x<d2.x) ? d1 : d2;
}

float opSubtraction( float d1, float d2 ) { return max(-d1,d2); }

vec4 tree1(vec3 p, float randValue,  mat3 rotationLeaf) {
  	vec4 trunc = vec4(sdCappedCylinder((( p + vec3(0., -.5, 0)) ) , vec2(0.15, 2.) * randValue) , TRUNK);
  	vec4 leaf = vec4(sdOctahedron(((rotationLeaf * p + vec3(0., -3. * randValue, 0.)) ) , 2.120  * randValue), vec3(0.915,0.191,0.094));           
  	return unionSDF(trunc, leaf);
}

vec4 tree2(vec3 p, float randValue,  mat3 rotationLeaf) {
 	vec4 trunc = vec4(sdCappedCylinder(((p + vec3(0., -.5, 0)) ) , vec2(0.5,2.990) * randValue), TRUNK);
	vec4 leaf = vec4(piramidSDF(rotationLeaf * p + vec3(0, -3.5 * randValue, 0.), vec3(1.8, 1., 1.2) * randValue), vec3(0.500,0.482,0.171));
  	return unionSDF(trunc, leaf);
}

vec4 tree3(vec3 p, float randValue, mat3 rotationLeaf) {
 	float scale = 1.1 * randValue;
  	vec4 trunc = vec4(sdCappedCylinder((((p + vec3(0., -1.5, 0)) ) ) , vec2(0.2,3.0) * scale), TRUNK);
	vec4 leaf = vec4(sdHexPrism(((rotationLeaf * p + vec3(0, -3.8 * scale, 0.))), vec2(1.5, 1.2) * scale ), vec3(0.500,0.414,0.075));
	return unionSDF(trunc, leaf);
}


vec4 tree4(vec3 p, float randValue, mat3 rotationLeaf) {
    float scale = 1.3 * randValue;
	vec4 trunc = vec4(sdCappedCylinder(((rotateX(PI) * p + vec3(0., -1.5, 0))), vec2(0.4,4.0) * scale), TRUNK);
	vec4 leaf = vec4(sdBox(((rotationLeaf  *  p + vec3(0., -4. *scale, 0.)) ) , vec3(1.5) * scale), vec3(0.690,0.411,0.121));
	return unionSDF(trunc, leaf);
}


vec3 pModXZ(vec3 p, const in vec3 size) {
  p.x = mod(p.x + size.x * 0.5, size.x) - size.x * 0.5;
  p.z = mod(p.z + size.z * 0.5, size.z) - size.z * 0.5;
  return p;
}

vec4 createTrees(vec3 samplePoint) {
    vec4 scene = vec4(1.);
    
    vec3 domainRepition = pModXZ(vec3(samplePoint.x - 2.5, samplePoint.y - 2.5, samplePoint.z + u_time  * SPEED), vec3(12.5, 0., 25. ));   

    vec3 tree1Repeat = domainRepition;
    vec3 tree2Repeat = vec3(tree1Repeat.x - 3.5 , tree1Repeat.y, tree1Repeat.z + 5.5 );;
    vec3 tree3Repeat = vec3(tree1Repeat.x - 2.5, tree1Repeat.y, tree1Repeat.z + 11.5);
    vec3 tree4Repeat = vec3(tree1Repeat.x - 1.3, tree1Repeat.y, tree1Repeat.z - 6.5);
    
    float scaleDistance = min(1., (1.2 + -samplePoint.z * 0.02));
    mat3 rotationLeaf = rotateY(PI * scaleDistance);
    
    vec4 tree1 = tree1(tree1Repeat, scaleDistance, rotationLeaf);
    vec4 tree2 = tree2(tree2Repeat, scaleDistance, rotationLeaf);
    vec4 tree3 = tree3(tree3Repeat, scaleDistance, rotationLeaf);
    vec4 tree4 = tree4(tree4Repeat, scaleDistance, rotationLeaf);

    scene = unionSDF(scene, tree1);
    scene = unionSDF(scene, tree2);
    scene = unionSDF(scene, tree3);
    scene = unionSDF(scene, tree4);

    return scene;
}


vec4 createCar(vec3 p) {
    float jumping = mix(0., .3, sin(u_time * 5.));
    
   	vec4 car = vec4(sdBox(p + vec3(0., -2. - jumping, 0), vec3(2., 2., 3.9)), vec3(0.170,0.274,0.325));
	float subFront = sdBox(  p + vec3(0., -3. - jumping, -3.5), vec3(2.5, 1.3, 1.2));
    float subBack = sdBox(  p + vec3(0., -3. - jumping, 3.5), vec3(2.5, 1.3, 1.2));
    
    car.x = opSubtraction(subFront, car.x);
    car.x = opSubtraction(subBack, car.x);
    
    vec4 windowBack =  vec4(sdBox(p + vec3(0., -3. - jumping, 2.3), vec3(1.3, .43, 0.01)) - 0.3, vec3(0.505,0.540,0.510));
    vec4 windowLeft =  vec4(sdBox(rotateY(-1.548) * p + vec3(0., -3. - jumping, 1.8), vec3(1.3, .43, 0.01)) - 0.3, vec3(0.505,0.540,0.510));
    car = unionSDF(car, windowBack);
    car = unionSDF(car, windowLeft);

    vec3 t =  rotateZ(1.564) * p;
   
    vec3 wheelBackPosition = t + vec3(-0.2 - jumping * .5, .4 , 2.1);
    vec3 wheelFrontPosition = t + vec3(-0.2 - jumping * .5, .4, -2.1);
    
    vec4 wheel = vec4(sdCappedCylinder(wheelBackPosition, vec2(1., 2.1)), CAR_TIRES);
    vec4 wheel2 = vec4(sdCappedCylinder(wheelFrontPosition, vec2(1., 2.2)), CAR_TIRES);
    
    vec4 wheelWhite = vec4(sdCappedCylinder(wheelBackPosition, vec2(.4, 2.1)), vec3(1.));
    vec4 wheelWhite2 = vec4(sdCappedCylinder(wheelFrontPosition, vec2(.4, 2.2)), vec3(1.));

    
    car = unionSDF(car, wheel);
    car = unionSDF(car, wheel2);
    
    car = unionSDF(car, wheelWhite);
    car = unionSDF(car, wheelWhite2);

    return car;
}

vec4 createFence(vec3 p) {
    const vec3 pillarColorMain = vec3(0.355,0.340,0.337);
    const vec3 pillarColorFence = vec3(0.255,0.243,0.248); 
        
    vec4 pillar = vec4(sdBox(p + vec3(TREES_ROAD_OFFSET_RIGHT - 2., -.5, 0), vec3(.15, 2., 100.)), pillarColorMain);
	vec4 fence = vec4(sdBox(p + vec3(TREES_ROAD_OFFSET_RIGHT - 2., -2.5, 0), vec3(.25, 0.12, 100.)), pillarColorFence);
    
    vec4 pillarLeft = vec4(sdBox(p + vec3(-TREES_ROAD_OFFSET_RIGHT - 2., -.5, 0), vec3(.15, 2., 100.)), pillarColorMain);
	vec4 fenceLeft = vec4(sdBox(p + vec3(-TREES_ROAD_OFFSET_RIGHT - 2., -2.5, 0), vec3(.25, 0.12, 100.)), pillarColorFence);

   if (mod(p.z + SPEED * u_time, SPEED) > .5) {
       pillar.x = 0.35;
       pillarLeft.x = 0.35;
    }
    
    vec4 fenceR = unionSDF(pillar, fence);
    vec4 fenceL = unionSDF(pillarLeft, fenceLeft);
    
    return unionSDF(fenceL, fenceR);
}

vec4 map(vec3 samplePoint) {
    vec4 plane = vec4(sdPlane(samplePoint), ROAD);
    if (mod(samplePoint.z + SPEED * u_time , 16.) > 6.600 && samplePoint.x < 0. && samplePoint.x > -0.7) {
        plane.yzw = vec3(1.0);
    }
    
    vec4 trees = createTrees(samplePoint);

    if (samplePoint.x < -ROAD_WIDTH || samplePoint.x > ROAD_WIDTH ) {
        plane.yzw = vec3(0.255,0.152,0.036);
    } else {
        trees.x = 1.;
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

    vec3 ro = vec3(6., 22., -13.6);
    
    vec3 ta =  normalize(vec3(0.,0.,-1.000));
    mat3 ca = calcLookAtMatrix(ro, ta, 0.0);
    vec3 rd = ca * normalize(vec3(p.xy, 1.2));
    
    vec3 color = vec3(1.0);
    vec4 scene = raymarsh(ro, rd, MIN_DIST, MAX_DIST);

   color = scene.yzw;

     vec3 point = ro + scene.x * rd;
     vec3 nor = getNormal(point);


    float shininess = 5.824;

    vec3 lightPosition =  vec3(-10.0, 20.0, 30.);
    color *= phongIllumination(vec3(1.8), vec3(2.5), vec3(1.100,0.893,0.064), shininess, point ,  ro, lightPosition);
    // color *= vec3(2.5);
    // color *= calcSoftshadow(p, vec3(0.3,3 , 1.));
    // color *= calcAO( p, nor );

    
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
    color = smoothstep(0., .7, color);
    gl_FragColor = vec4(color, 1.0);
}
